"""
This file wraps all necessary models in rlhf with ray actor
1. openrlhf ray launcher.py
"""

# Reward-model actors used during RLHF rollouts to score generations. Two
# families are implemented: rule-based rewards (CPU-only, e.g. keyword or
# groundtruth checks) and generative reward models served via vLLM (a
# learned "forward" reward model is stubbed out but not implemented). When
# ``cot_end_token`` is configured (RECAP counter-aligned CoT prefilling), it
# is used to strip the injected CoT prefix from the generation text before
# scoring, so reward reflects only the model's own response.

import logging
from typing import Any, Callable, Dict, List, Union

import ray
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from openrlhf.utils.deepspeed import DeepspeedStrategy
from ray.util.placement_group import PlacementGroup
from vllm import SamplingParams
from vllm.outputs import RequestOutput

from src.data.util import detect_message_type

from .ray_actor_group import BaseModelActor
from .util import get_tokenizer
from .vllm_engine import LLMRayActor, create_vllm_engines

logger = logging.getLogger(__name__)


__all__ = [
    "RuleRewardModel",
    "RuleRewardModelActor",
    "GenerativeRewardModelActorGroup",
]

# === All RM should take formatted conversation as input and return a float scalar, the experience should format the chat before sending to RM
# for RM naming, we will use "<domain>_<subdomain>_<method>", e.g., safety_keyword, safety_guardrail, math_gsm8k_rule

# rule based RM, for safety, we use keyword detection, for math, we need the groundtruth label


class RuleRewardModel(nn.Module):
    def __init__(
        self,
        rule_fn: Callable[[str], float],
    ):
        """
        rule based RM may not care prompt, directly takes response and do downstream checks
        Args:
            rule_fn: A function that takes a string (decoded text) and returns a float reward.
            tokenizer: Tokenizer used to decode input_ids.
            input_type: Either "response" (decode the output only) or "full" (decode entire prompt + output).
        """
        super().__init__()
        self.rule_fn = rule_fn

    def forward(
        self,
        conversations: List[List[Dict[str, str]]],
        generation_texts: List[str],
    ) -> torch.Tensor:
        """
        depending on the type of rules we will use here, we will prepare the corresponding conversation here
        we will send conversation, and generation text to all rule fn, and rule fn decides whether to use it or not
        Args:
            input_ids: [B, L] tensor of token ids
            prompt_lengths: Optional list of ints indicating prompt lengths (used if input_type='response')
        Returns:
            rewards: [B] float tensor of scalar rewards
        """
        rewards = self.rule_fn(
            conversations=conversations, generations=generation_texts
        )

        assert len(rewards) == len(generation_texts) == len(conversations)

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards


class ForwardRewardModel(nn.Module):
    """Placeholder for a learned (model-based) reward model; not yet implemented."""

    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, model_cfg: DictConfig, train_cfg: DictConfig
    ):
        self._setup_distributed(strategy)
        raise NotImplementedError


@ray.remote
class RuleRewardModelActor(BaseModelActor):
    """rule RM actor only needs cpu, i.e., no deepspeed strategy is needed here"""

    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, model_cfg: DictConfig, **kwargs: Any
    ):
        """Instantiate the configured rule function (via Hydra) as a RuleRewardModel."""
        self.strategy = strategy
        rule_fn = instantiate(model_cfg.rule_fn)
        self.model = RuleRewardModel(rule_fn)

        # additional args for RM
        self.cot_end_token = kwargs.get("cot_end_token", None)
        self.use_overlong_filtering = kwargs.get("use_overlong_filtering", False)

    def forward(
        self,
        conversations: List[List[Dict[str, str]]],
        generation_texts: List[str],
    ) -> torch.Tensor:
        """Score generations with the rule function, stripping the CoT prefix first if configured."""
        # drop cot if exists
        missing_cot_idxs = None
        if self.cot_end_token:
            if self.use_overlong_filtering:
                # generations without the CoT end token likely got truncated
                # (ran out of budget) before finishing the CoT
                missing_cot_idxs = [
                    i
                    for i, t in enumerate(generation_texts)
                    if self.cot_end_token not in t
                ]
            # if cot is missing, we will pass the cot into the reward
            generation_texts = [
                text.split(self.cot_end_token)[-1] for text in generation_texts
            ]

        rewards = self.model(
            conversations=conversations, generation_texts=generation_texts
        )
        if missing_cot_idxs:
            print(f"inside rule RM actor: {len(missing_cot_idxs)}")

        return rewards


# This is working at the same level as the RayActorGroup, as we dont need deepspeed strategy for generative RMs
class GenerativeRewardModelActorGroup:
    """generative RM actor needs gpu
    this is not based on BaseModelActor as vllm performs the batch call
    """

    def __init__(
        self,
        vllm_num_engines: int,
        vllm_tensor_parallel_size: int,
        train_cfg: DictConfig,
        model_cfg: DictConfig,
        seed: int,
        full_determinism: bool,
        pg: PlacementGroup = None,
        **kwargs: Any,
    ) -> None:
        """Spin up vLLM engines and chat-template/sampling config for a generative judge model."""
        self.rm_type = "generative"
        self.model_name = model_cfg.name.split("/")[-1].lower()
        self.max_length = model_cfg.prompt_max_len + model_cfg.generate_max_len

        # vllm LLM
        self.llms = create_vllm_engines(
            num_engines=vllm_num_engines,
            tensor_parallel_size=vllm_tensor_parallel_size,
            pretrain=model_cfg.name,
            seed=seed,
            full_determinism=full_determinism,
            enable_prefix_caching=model_cfg.enable_prefix_caching,
            enforce_eager=False,
            max_model_len=self.max_length,
            shared_pg=None,
            gpu_memory_utilization=train_cfg.vllm_gpu_memory_utilization,
            vllm_enable_sleep=train_cfg.vllm_enable_sleep,
            llm_actor_cls=LLMRayActor,
        )

        # load tokenizer
        tokenizer = get_tokenizer(
            pretrain=model_cfg.name,
            model=None,
            padding_side="left",
            use_fast=not model_cfg.disable_fast_tokenizer,
        )
        if model_cfg.chat_template is not None:
            logger.info("Overwritting generative reward model's chat template ...")
            tokenizer.chat_template = model_cfg.chat_template
        self.tokenizer = tokenizer

        chat_template_kwargs = (
            {} if model_cfg.template_args is None else model_cfg.template_args
        )
        chat_template_kwargs["continue_final_message"] = False
        chat_template_kwargs["add_generation_prompt"] = True
        self.chat_template_kwargs = chat_template_kwargs
        self.to_message = detect_message_type(tokenizer.chat_template)

        # vllm sampling params
        g_args = model_cfg.generation_args
        self.sampling_params = SamplingParams(
            max_tokens=model_cfg.generate_max_len,
            top_p=g_args.top_p,
            top_k=g_args.top_k,
            temperature=g_args.temperature,
            repetition_penalty=g_args.repetition_penalty,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            logprobs=10,  # so there may be up to logprobs+1 elements in the response.
        )

        # additional args for RM
        self.cot_end_token = kwargs.get("cot_end_token", None)

        # output to reward
        self.output_to_reward_fn = instantiate(
            model_cfg.output_to_reward_fn,
            model_name=self.model_name,
            tokenizer=tokenizer,
        )

    def forward(
        self,
        conversations: List[List[Dict[str, str]]],
        generation_texts: List[str],
    ) -> List[float]:
        """Append generations to their conversations, dispatch to vLLM engines, and return response refs.

        Returns:
            List of per-engine remote object refs; resolve with :meth:`get_reward`
            after retrieving them (e.g. via ``ray.get``).
        """
        assert len(conversations) == len(generation_texts), (
            f"{len(conversations)} conversations not matching {len(generation_texts)} generations"
        )

        # drop cot if exists
        if self.cot_end_token:
            generation_texts = [
                text.split(self.cot_end_token)[-1] for text in generation_texts
            ]

        reward_input_tokens = []
        for idx, generation_text in enumerate(generation_texts):
            assert conversations[idx][0]["role"] in ["user", "system"]

            last_role = conversations[idx][-1]["role"]
            if last_role == "assistant":
                conversations[idx][-1]["content"] = generation_text
            else:
                conversations[idx].append(
                    self.to_message(role="assistant", content=generation_text)
                )

            conversation_text = self.tokenizer.apply_chat_template(
                conversations[idx],
                tokenize=False,
                **self.chat_template_kwargs,
            )

            reward_input_token = self.tokenizer(
                conversation_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                add_special_tokens=False,
                return_length=False,
            )

            reward_input_tokens.append(reward_input_token["input_ids"])

        # model generation, distribute requests to engines
        refs = []
        batch_size = (len(reward_input_tokens) + len(self.llms) - 1) // len(self.llms)
        for i, llm in enumerate(self.llms):
            prompt_token_ids = reward_input_tokens[
                i * batch_size : (i + 1) * batch_size
            ]
            refs.append(
                llm.add_requests.remote(
                    sampling_params=self.sampling_params,
                    prompt_token_ids=prompt_token_ids,
                )
            )
        ray.get(refs)

        # Retrieve and combine results from all outputs
        output_refs = []
        for i, llm in enumerate(self.llms):
            output_refs.append(llm.get_responses.remote())

        return output_refs

    def get_reward(
        self, generation: Union[List[RequestOutput], List[List[RequestOutput]]]
    ):
        """Convert resolved vLLM outputs from :meth:`forward` into a scalar reward tensor."""
        # convert model outputs to scalar reward
        rewards = self.output_to_reward_fn(generations=generation)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards

    def empty_cache():
        # send vllm to sleep, only called when we colocate models
        pass
