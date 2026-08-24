"""Rollout generation and experience construction for online RLHF.

Analogous to OpenRLHF's ``ppo_utils/experience_maker.py``. Two collaborators:

* :class:`SamplesGenerator` — samples completions from the vLLM engines for a
  batch of (possibly prefilled) prompts and packages each into an
  :class:`~src.data.Experience` with ``sequences``, ``attention_masks``, and an
  ``action_mask`` that marks only the response region. Because the tokenized
  prompt already contains any counter-aligned prefill, this action mask is what
  restricts RECAP's optimization to the tokens the policy generates itself.
* :class:`ExperienceMaker` — runs the policy/reference/reward/critic forward
  passes over those samples to fill in log-probs, KL, rewards, and (group-)
  normalized advantages before the policy update.
"""

import itertools
import logging
import time
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import ray
import torch
from omegaconf import DictConfig
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from ray.actor import ActorHandle
from transformers import AutoTokenizer
from vllm import SamplingParams

from src.data import Experience
from src.model import RayActorGroup, batch_vllm_engine_call

from .util import compute_overlong_reward, flatten_tensor_dict, resolve_dict_futures

__all__ = ["ExperienceMaker", "SamplesGenerator"]

logger = logging.getLogger(__name__)


class SamplesGenerator:
    """rollout and online sampling"""

    def __init__(
        self,
        vllm_engines: List[ActorHandle],
        tokenizer: AutoTokenizer,
        generation_cfg: DictConfig,
        train_cfg: DictConfig,
        apply_chat_template: bool = True,
        remote_reward_model: Optional = None,
    ):
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.remote_reward_model = remote_reward_model

        self.g_cfg = generation_cfg
        self.train_cfg = train_cfg
        self.vllm_enable_sleep = train_cfg.vllm_enable_sleep
        self.prompt_max_len = train_cfg.prompt_max_len
        self.generate_max_len = train_cfg.generate_max_len
        self.n_samples_per_prompt = train_cfg.n_samples_per_prompt
        self.enable_overlong_punishment = train_cfg.overlong_buffer.enable
        self.overlong_buffer_len = train_cfg.overlong_buffer.len

        # create a counter for experience that starts at 0 and increments by 1 each time
        self._rollout_id_counter = itertools.count()

    @torch.no_grad()
    def generate_samples(
        self,
        all_tokenized_prompts: List[List[int]],
        info: Dict[str, List],
    ) -> List[Experience]:
        """
        Generate samples and return in batches.
        """
        start_time = time.time()
        logger.info(
            f"==== Starting generating samples: {len(all_tokenized_prompts)} prompts * {self.n_samples_per_prompt} rollouts each ===="
        )
        # vLLM wakeup when vllm_enable_sleep
        if self.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        self.data_info = info
        rollout_samples = self._generate_vllm(all_tokenized_prompts)

        # vLLM offload when vllm_enable_sleep
        if self.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"==== Rollout completed in {time_str} (HH:MM:SS) ====")

        return rollout_samples

    def _generate_vllm(
        self, all_tokenized_prompts: List[List[int]]
    ) -> List[Experience]:
        """
        because we have tokenized in dataloader, so we don't need to tokenize again here, and we drop the padding in collate fn
        Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to , we don't use label for online sampling, in the future we may consider prefilling somewhere
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """

        llms = self.vllm_engines

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=self.g_cfg.get("temperature", 1.0),
            top_p=self.g_cfg.get("top_p", 1.0),
            top_k=self.g_cfg.get("top_k", -1),
            max_tokens=self.generate_max_len,  # during training, max_new_tokens is determined by cfg.train not cfg.model
            min_tokens=self.g_cfg.get("min_new_tokens", 1),
            skip_special_tokens=self.g_cfg.get("skip_special_tokens", True),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompt_token_ids = [
            prompt
            for prompt in all_tokenized_prompts
            for _ in range(self.n_samples_per_prompt)
        ]
        all_reward_names = [
            reward
            for reward in self.data_info["reward"]
            for _ in range(self.n_samples_per_prompt)
        ]
        all_conversations = [
            conversation
            for conversation in self.data_info["conversation"]
            for _ in range(self.n_samples_per_prompt)
        ]

        # Distribute requests to engines and collect responses
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[
                i * batch_size : (i + 1) * batch_size
            ]
            refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params, prompt_token_ids=prompt_token_ids
                )
            )
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        # create experience from vllm outputs
        truncate_length = self.prompt_max_len + self.generate_max_len
        eos_token_id = self.tokenizer.eos_token_id

        samples_list = []
        for conversation, output, reward_name in zip(
            all_conversations, all_outputs, all_reward_names
        ):
            # extract token IDs, the prompt_ids also include tokens for prefilling
            prompt_ids = list(output.prompt_token_ids)
            response_ids = list(output.outputs[0].token_ids)
            generation_text = output.outputs[0].text

            # soft overlong punishment
            if self.enable_overlong_punishment:
                expected_len = self.generate_max_len - self.overlong_buffer_len
                exceed_len = len(response_ids) - expected_len

            # ensure EOS is present
            if response_ids[-1] != eos_token_id:
                response_ids.append(eos_token_id)

            # construct sequence and attention mask
            input_ids = prompt_ids + response_ids
            sequences = torch.tensor(input_ids[:truncate_length], dtype=torch.long).to(
                "cpu"
            )
            attention_mask = torch.ones_like(sequences)

            # Create action mask (response-only region). This is exactly where
            # RECAP's prefix masking happens for free: because prompt_ids above
            # includes the counter-aligned CoT prefill, response_start points
            # *after* the prefix, so only the policy-generated tokens (t > ℓ_pre)
            # are marked as actions and enter the loss / advantage computation.
            action_mask = torch.zeros_like(attention_mask)
            response_start = len(prompt_ids)
            response_end = min(response_start + len(response_ids), truncate_length)
            action_mask[response_start:response_end] = 1
            action_mask = action_mask[1:]  # Drop BOS token

            # meta info
            info = {
                "response_length": torch.tensor([len(response_ids)]),
                "total_length": torch.tensor([len(sequences)]),
            }

            rollout_samples = Experience(
                rollout_ids=[next(self._rollout_id_counter)],
                conversations=[conversation],
                generation_texts=[generation_text],
                reward_names=[reward_name],
                exceed_lens=[exceed_len] if self.enable_overlong_punishment else None,
                sequences=sequences.unsqueeze(0),
                attention_masks=attention_mask.unsqueeze(0),
                action_masks=action_mask.unsqueeze(0),
                info=info,
            )
            samples_list.append(rollout_samples)

        # This is required by dynamic sampling
        if self.remote_reward_model:
            raise NotImplementedError

        return samples_list


class ExperienceMaker(ABC):
    def __init__(
        self,
        policy_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        multi_reward_model_group: List[RayActorGroup],
        ref_model_group: RayActorGroup,
        policy_tokenizer: AutoTokenizer,
        kl_controller,
        train_cfg: DictConfig = None,
        ring_attn_size: int = 1,
        ds_tensor_parallel_size: int = 1,
        remote_reward_model: Optional = None,
    ):
        super().__init__()

        self.policy_model_group = policy_model_group
        self.critic_model_group = critic_model_group
        self.multi_reward_model_group = multi_reward_model_group
        self.ref_model_group = ref_model_group
        self.policy_tokenizer = policy_tokenizer
        self.train_cfg = train_cfg
        self.remote_reward_model = remote_reward_model
        self.ring_attn_size = ring_attn_size
        self.ds_tensor_parallel_size = ds_tensor_parallel_size
        self.kl_ctl = kl_controller

        # below are the hyperparams from train cfg
        self.advantage_estimator = train_cfg.advantage_estimator
        self.colocate_all_models = train_cfg.colocate_all_models
        self.colocate_actor_ref = train_cfg.colocate_actor_ref
        self.use_kl_loss = train_cfg.use_kl_loss
        self.kl_estimator = train_cfg.kl_estimator
        self.n_samples_per_prompt = train_cfg.n_samples_per_prompt
        self.reward_clip_range = train_cfg.reward_clip_range
        self.reward_discount = train_cfg.reward_discount
        self.enable_overlong_punishment = train_cfg.overlong_buffer.enable
        self.overlong_buffer_len = train_cfg.overlong_buffer.len
        self.overlong_penalty_factor = train_cfg.overlong_buffer.penalty_factor

    @torch.no_grad()
    def make_experience_batch(
        self, rollout_samples: List[Experience]
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        start_time = time.time()
        self.total_experiences = len(rollout_samples)
        logger.info(
            f"==== Start making experiences with {self.total_experiences} samples ===="
        )

        # 1. send the non-concat rollout_samples to the generative RM and extract reward for generative-only
        self.make_generative_reward(rollout_samples)

        # 2. send the concat rollout_samples to rule/forward RM & policy & ref (all generative reward samples will pass through non-generative reward due to balancing the ray hang on uneven batches)
        # Concat samples into micro_rollout_batch_size; each batch is assigned to a Ray Actor
        experiences_list = []
        self.batch_size = self.train_cfg.micro_rollout_batch_size
        for i in range(0, len(rollout_samples), self.batch_size):
            concat_samples = Experience.concat_experiences(
                rollout_samples[i : i + self.batch_size],
                self.policy_tokenizer.pad_token_id,
            )
            experiences_list.append(concat_samples)
        experiences_list = self.make_non_generative_reward_and_logprob(experiences_list)

        # 3. compute advantanges and returns
        experiences_list = self.compute_advantages_and_returns(experiences_list)

        # 4. split experiences_list into list of individual experience
        experiences_list = [
            exp
            for exp_batch in experiences_list
            for exp in Experience.split_experience_batch(exp_batch)
        ]

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"==== Experience making completed in {time_str} (HH:MM:SS) ====")

        return experiences_list

    @torch.no_grad()
    def make_generative_reward(
        self, rollout_samples: List[Experience]
    ) -> Dict[str, List]:
        """
        compute reward for rollout_samples that need generative reward
        we will only return the reward, and modidy the experience later as this avoids running ray sequentially
        """
        # create reward jobs for all generative reward models
        jobs: Dict[str, List[Experience]] = defaultdict(list)
        for exp in rollout_samples:
            jobs[exp.reward_names[0]].append(exp)

        # the order in samples_rewards is preserved
        generative_reward_ids: Dict[str, List[int]] = {}
        generative_reward_jobs: Dict[str, List[int]] = {}
        for name, exps in jobs.items():
            reward_model, reward_model_cfg = self.multi_reward_model_group[name]
            combined_exps = Experience.concat_experiences(
                exps, self.policy_tokenizer.pad_token_id
            )
            if reward_model_cfg.type == "generative":
                reward_refs = reward_model.forward(
                    conversations=combined_exps.conversations,
                    generation_texts=combined_exps.generation_texts,
                )
            else:
                reward_refs = ray.put([[None]] * len(exps))
            generative_reward_ids[name] = combined_exps.rollout_ids
            generative_reward_jobs[name] = reward_refs

        self.generative_reward_ids = generative_reward_ids
        self.generative_reward_jobs = generative_reward_jobs

    def merge_and_sort_generative_rewards(
        self, generative_reward_ids: dict, generative_rewards: dict
    ) -> list[float]:
        # Step 1: Globally collect all (id, reward) pairs
        all_pairs = []
        for rm_name in generative_reward_ids:
            ids = generative_reward_ids[rm_name]
            rewards = generative_rewards[rm_name]
            assert len(ids) == len(rewards), f"Mismatch in lengths for {rm_name}"
            all_pairs.extend(zip(ids, rewards))

        # Step 2: Sort all pairs by ID
        all_pairs.sort(key=lambda x: x[0])

        # Step 3: Fill in final reward list in ascending ID order
        final_rewards = [reward for _, reward in all_pairs]

        # Sanity check
        assert len(final_rewards) == self.total_experiences, (
            f"Expected {self.total_experiences} rewards, got {len(final_rewards)}"
        )

        return final_rewards

    @torch.no_grad()
    def make_non_generative_reward_and_logprob(
        self, experiences_list: List[Experience]
    ) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        when we colocate models, we will run each player in rl sequentially, otherwise ray.get will only be called at the end
        when calling at the end, we shall do xx_ref, xx_ref = ray.get([xx_ref, xx_ref]) dont call it sequentially as the blocking is not parallel
        """
        # Extract all information from samples in one pass
        # policy & ref
        sequences_list = [exp.sequences for exp in experiences_list]
        attention_masks_list = [exp.attention_masks for exp in experiences_list]
        action_masks_list = [exp.action_masks for exp in experiences_list]
        # reward model
        conversations_list = [exp.conversations for exp in experiences_list]
        generation_texts_list = [exp.generation_texts for exp in experiences_list]
        # soft overlong punishment
        if self.enable_overlong_punishment:
            exceed_lens_list = [exp.exceed_lens for exp in experiences_list]

        n = len(experiences_list) * self.ring_attn_size * self.ds_tensor_parallel_size

        # ==== reward
        if experiences_list[0].rewards is not None:
            pass
        elif self.remote_reward_model is not None:
            raise NotImplementedError
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(
                        remove_pad_token(seq, attention_mask), skip_special_tokens=False
                    )
                    for seq, attention_mask in zip(sequences_list, attention_masks_list)
                ],
                [],
            )
            prompts_list = sum([s.prompts for s in experiences_list], [])
            labels_list = sum([s.labels for s in experiences_list], [])
            # Keep the remote call asynchronous
            r_refs = self.remote_reward_model.get_rewards.remote(
                queries_list, prompts_list, labels_list
            )
        else:
            # Batch call reward model
            non_generative_reward_jobs = {}
            for reward_name, (
                reward_model,
                reward_model_cfg,
            ) in self.multi_reward_model_group.items():
                if reward_model_cfg.type != "generative":
                    reward_refs = reward_model.async_run_method_batch(
                        method_name="forward",
                        conversations=conversations_list,
                        generation_texts=generation_texts_list,
                    )
                    non_generative_reward_jobs[reward_name] = reward_refs

        # TODO, fix reward model when colocating models
        # Sync to avoid GPU OOM when colocate models
        if self.colocate_all_models and self.remote_reward_model is None:
            raise NotImplementedError
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        action_log_probs_ref = self.policy_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_masks=action_masks_list,
            attention_masks=attention_masks_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if self.colocate_all_models or self.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.policy_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if self.colocate_critic_reward and not self.remote_rm_url:
                raise NotImplementedError
                ray.get(r_refs)
                ray.get(
                    self.reward_model_group.async_run_method(method_name="empty_cache")
                )

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_masks=action_masks_list,
                attention_masks=attention_masks_list,
            )
            if self.colocate_all_models or self.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(
                    self.critic_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            value_ref = ray.put([[None]] * n)

        # Batch call reference model
        if self.ref_model_group is not None:
            base_action_log_probs_ref = self.ref_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_masks=action_masks_list,
                attention_masks=attention_masks_list,
            )

            if self.colocate_all_models or self.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(
                    self.ref_model_group.async_run_method(method_name="empty_cache")
                )
        else:
            base_action_log_probs_ref = ray.put([[None]] * n)

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = self.ring_attn_size * self.ds_tensor_parallel_size
        action_log_probs_list = [
            item
            for sublist in ray.get(action_log_probs_ref)[::duplicate_factor]
            for item in sublist
        ]

        base_action_log_probs_list = [
            item
            for sublist in ray.get(base_action_log_probs_ref)[::duplicate_factor]
            for item in sublist
        ]
        value_list = [
            item
            for sublist in ray.get(value_ref)[::duplicate_factor]
            for item in sublist
        ]

        # Process rewards based on source
        if experiences_list[0].rewards is not None:
            pass
        elif self.remote_reward_model is not None:
            raise NotImplementedError
            # Get rewards info from remote model
            rewards_info = ray.get(r_refs)
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, experiences_list)
        else:
            generative_rewards = resolve_dict_futures(self.generative_reward_jobs)
            non_generative_rewards: Dict[str, List[torch.Tensor]] = (
                resolve_dict_futures(non_generative_reward_jobs)
            )

            # process generative rewards
            for name, generation in generative_rewards.items():
                reward_model, reward_model_cfg = self.multi_reward_model_group[name]
                if reward_model_cfg.type == "generative":
                    generative_rewards[name] = reward_model.get_reward(
                        generation=generation
                    )
            flattened_generative_rewards = self.merge_and_sort_generative_rewards(
                generative_reward_ids=self.generative_reward_ids,
                generative_rewards=generative_rewards,
            )
            flattened_non_generative_rewards = flatten_tensor_dict(
                non_generative_rewards
            )

            # soft overlong penalty
            if self.enable_overlong_punishment:
                overlong_rewards = [
                    compute_overlong_reward(
                        exceed_lens=exceed_lens,
                        buffer_len=self.overlong_buffer_len,
                        penalty_factor=self.overlong_penalty_factor,
                    )
                    for exceed_lens in exceed_lens_list
                ]
                print(overlong_rewards)

            # add reward to experiences
            for i, exp in enumerate(experiences_list):
                batched_rewards = []
                for j, reward_name in enumerate(exp.reward_names):
                    pos = i * self.batch_size + j
                    reward_type = self.multi_reward_model_group[reward_name][1].type
                    if reward_type == "generative":
                        reward_score = flattened_generative_rewards[pos]
                    else:
                        reward_score = flattened_non_generative_rewards[reward_name][
                            pos
                        ]
                    assert reward_score is not None
                    batched_rewards.append(reward_score)
                if self.enable_overlong_punishment:
                    batched_rewards = [
                        raw_r + len_r
                        for raw_r, len_r in zip(batched_rewards, overlong_rewards[i])
                    ]
                exp.rewards = torch.tensor(batched_rewards)
                exp.info["reward"] = exp.rewards

        assert (
            len(experiences_list)
            == len(action_log_probs_list)
            == len(base_action_log_probs_list)
            == len(value_list)
        ), (
            f"len(samples_list): {len(experiences_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"
        )

        # Process results for each sample
        for i, (exp, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(
                experiences_list,
                action_log_probs_list,
                base_action_log_probs_list,
                value_list,
            )
        ):
            # this is how grpo kl loss is excluded from the advantage computation, set use_kl_loss=true, and the kl will be deferred to the policy update
            if (self.ref_model_group is not None) and (not self.use_kl_loss):
                kl = compute_approx_kl(
                    log_probs=action_log_probs,
                    log_probs_base=base_action_log_probs,
                    kl_estimator=self.kl_estimator,
                )
            else:
                kl = torch.zeros_like(
                    action_log_probs, dtype=action_log_probs.dtype, device="cpu"
                )

            # return (B,) tensor - mean per online sampling
            kl_mean = masked_mean(kl, exp.action_masks, dim=-1)

            if not self.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            exp.action_log_probs = action_log_probs
            exp.base_action_log_probs = base_action_log_probs
            exp.values = value
            exp.kl = kl
            exp.info["kl"] = kl_mean

        return experiences_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences_list: List[Experience]
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """

        # get rewards from experiences
        rewards = [exp.rewards for exp in experiences_list]
        rewards = torch.cat(rewards).reshape(-1, self.n_samples_per_prompt)

        # record group reward std
        if self.n_samples_per_prompt > 1:
            std_per_prompt = rewards.std(-1, keepdim=True)
            # expand the std_per_prompt the same shape as rewards, and each chunk is std of rollout of each prompt
            group_reward_stds = (
                std_per_prompt.repeat(1, self.n_samples_per_prompt)
                .reshape(-1)
                .chunk(len(experiences_list))
            )
            for exp, group_reward_std in zip(experiences_list, group_reward_stds):
                exp.info["group_reward_std"] = group_reward_std

        # reward shaping
        if self.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (
                self.n_samples_per_prompt - 1
            )
            rewards = rewards - baseline
        elif self.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif self.advantage_estimator == "grpo":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (
                rewards.std(-1, keepdim=True) + 1e-9
            )

        # rewards: Tuple[Tensor]
        rewards = rewards.reshape(-1).chunk(len(experiences_list))

        # calculate return and advantages
        for exp, reward in zip(experiences_list, rewards):
            # this step injects the reward at the last token position, and add kl for ppo; for grpo, kl = 0 here
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                exp.kl,
                action_mask=exp.action_masks,
                reward_clip_range=self.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                raise NotImplementedError
                exp.advantages, exp.returns = self.get_advantages_and_returns(
                    exp.values,
                    reward,
                    exp.action_mask,
                    self.reward_discount,
                    self.lambd,
                )
            elif self.advantage_estimator in [
                "reinforce",
                "rloo",
                "reinforce_baseline",
                "grpo",
                "dr_grpo",
            ]:
                if self.reward_discount != 1.0 and self.advantage_estimator not in [
                    "reinforce"
                ]:
                    logger.warning(
                        "reward_discount is set to 1.0 for rloo, reinforce_baseline, and grpo"
                    )
                    self.reward_discount = 1.0

                exp.returns = self.compute_discounted_sequence_returns(
                    reward,
                    exp.action_masks,
                    self.reward_discount,
                )
                exp.advantages = deepcopy(exp.returns)
            else:
                raise Exception(
                    f"Unkown advantage_estimator {self.advantage_estimator}"
                )

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            exp.info["return"] = return_sums
            # remove unnecessary info
            exp.kl = None

        # Normalize advantages across all experiences
        if self.advantage_estimator not in ["grpo", "dr_grpo"]:
            raise NotImplementedError
            all_advantages = []
            all_action_masks = []
            for exp in experiences_list:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = (
                    (advantages_vector - mean).pow(2) * action_masks_vector
                ).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences_list:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences_list

    @torch.no_grad()
    def compute_discounted_sequence_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        reward_discount: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - reward_discount: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + reward_discount * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        reward_discount: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This func is gae for ppo training"""
        raise NotImplementedError
