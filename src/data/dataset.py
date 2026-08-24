"""Hydra-configured dataset for SFT, RLHF rollouts, inference, and evaluation.

A single :class:`HydraConfigDataset` serves every stage; two flags select the
mode:

* ``use_response`` — whether a ground-truth/label response is tokenized together
  with the prompt (True for SFT and guardrail scoring; False for RLHF and plain
  inference, which tokenize the prompt only).
* ``use_prefill`` — whether a **counter-aligned CoT prefill** (the RECAP
  mechanism) is injected. When set, the ``prefill`` column is appended as the
  start of the assistant turn and the chat template is rendered with
  ``continue_final_message=True`` so the model continues *from* that flawed
  reasoning. Crucially, ``prompt_ids_len`` then counts prompt **+ prefill**
  tokens, marking the boundary (ℓ_pre in the paper) after which the policy's
  generated tokens begin — the RL objective is later masked to optimize only
  those post-prefill tokens.

Mode matrix (flag combinations actually used):
    inference:              (—)                 prompt only
    inference + prefill:    (use_prefill)       prompt + injected CoT prefix
    SFT / guardrail:        (use_response)      prompt + response
    SFT/eval + prefill:     (use_prefill, use_response)
"""

import logging
from typing import Tuple, Union

import datasets
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .util import (
    can_print,
    detect_message_type,
    format_prompt_and_response,
    load_system_prompt_from_file,
)

logger = logging.getLogger(__name__)


class HydraConfigDataset(Dataset):
    """
    Dataset for SFT and RLHF using hydra config

    Args:
        data: dataset for SFT & RLHF
        tokenizer: tokenizer for SFT model
        max_length: max length of input

        if set use_response=true, we will apply chat template on both prompt + response. This can be used for SFT, any prefilling, or guardrail evaluation on full response
        if set to false, this is for non-prefilling, this can be used for normal batch_eval and rlhf with prompt only
    """

    def __init__(
        self,
        data: datasets.Dataset,
        data_cfg: DictConfig,
        model_cfg: DictConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int,
        num_processors: int = 8,  # Specify the number of processors you want to use
        multiturn: bool = False,
        pretrain_mode: bool = False,
        for_guardrail: bool = False,
        use_response: bool = True,
        use_value: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length  # note the max length here is to truncate the tokenized sequence, if use_response=true, then it's length of both prompt and response, if false, then prompt only
        self.pretrain_mode = pretrain_mode
        self.for_guardrail = for_guardrail  # drop system prompt if for_guardrail=True
        self.use_response = use_response
        self.use_value = use_value
        self.data_cfg = data_cfg
        self.apply_chat_template = data_cfg.apply_chat_template
        self.use_prefill = data_cfg.use_prefill
        self.system_prompt = load_system_prompt_from_file(model_cfg.system)
        self.to_message = detect_message_type(tokenizer.chat_template)

        # ensure correct template
        chat_template_kwargs = (
            {} if model_cfg.template_args is None else model_cfg.template_args
        )
        # either sft or guardrail needs to tokenize prompt and response, prompt guard will not set use_response
        if use_response:
            if for_guardrail:
                if can_print():
                    logger.info(
                        "setting add_generation_prompt=True for guardrail models"
                    )
                chat_template_kwargs["continue_final_message"] = False
                chat_template_kwargs["add_generation_prompt"] = True
            else:
                if can_print():
                    logger.info(
                        "setting both add_generation_prompt=False and continue_final_message=False for SFT"
                    )
                chat_template_kwargs["continue_final_message"] = False
                chat_template_kwargs["add_generation_prompt"] = False
        else:
            # normal inference, prefilling will overwrite the config again
            if can_print():
                logger.info(
                    "setting add_generation_prompt=True for inference, prefilling will overwrite this"
                )
            chat_template_kwargs["continue_final_message"] = False
            chat_template_kwargs["add_generation_prompt"] = True

        self.chat_template_kwargs = chat_template_kwargs

        self.multiturn = multiturn  # pack multiple sequences in one input

        # data processing
        processed_dataset = data.map(
            self.process_data,
            remove_columns=data.column_names,
            num_proc=num_processors,
            keep_in_memory=True,  # this is critical as the cache file name seems to be random and will lead to no such file error
            load_from_cache_file=False,
        )
        num_hit = sum(processed_dataset["hit_max_length"])
        if num_hit > 0 and can_print():
            logger.warning(
                f"{num_hit} samples have hit the max token length, increase prompt length for inferece and rlhf, and generation length for sft."
            )
        else:
            logger.info("All samples in dataset are below the max token length")

        self.model_inputs = processed_dataset["model_input"]
        self.conversations = processed_dataset["conversation"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        dataset_example = f"Dataset Example:\n{self.model_inputs[0]}"

        self.rewards = processed_dataset["reward"]
        if self.rewards[0] is not None:
            dataset_example += f"\nReward Type: {self.rewards[0]}"

        self.response_ranges = None
        if self.multiturn:
            self.response_ranges = processed_dataset["response_ranges"]
        if self.use_value:
            self.values = processed_dataset["value"]
            self.positions = processed_dataset["position"]

        if can_print():
            logger.info(dataset_example)

    def process_data(self, data: datasets.Dataset):
        # https://github.com/OpenRLHF/OpenRLHF/blob/806d0296aac32135f011edc00e1e95458e7e5095/openrlhf/datasets/sft_dataset.py#L91
        response_ranges = None
        if self.multiturn:
            raise NotImplementedError

        # value sft
        value = None
        position = None
        if self.use_value:
            value = data["value"]
            position = data["position"]
            assert len(value) == len(position), "Mismatch shape"

        # rename to prompt_message, prompt_text, response_message, response_text, prefill_message, prefill_text, conversation_message, conversation_text
        # processed prompt and response and return in message format
        prompt_message, response_message = format_prompt_and_response(
            data,
            self.apply_chat_template,
            chat_template=self.tokenizer.chat_template,
            for_guardrail=self.for_guardrail,  # if guardrail, strip system message
        )

        # apply chat template and combine prompt with response
        has_prefill_col = "prefill" in data and data["prefill"] is not None
        prompt_text = None
        if self.apply_chat_template:
            assert isinstance(prompt_message, list)

            # overwrite system prompt if provided externally
            if self.system_prompt is not None and not self.for_guardrail:
                system_message = self.to_message(
                    role="system", content=self.system_prompt
                )
                first_role = prompt_message[0]["role"]
                if first_role == "system":
                    prompt_message[0] = system_message
                else:
                    prompt_message.insert(0, system_message)

            # label provided (sft - no prefilling, but needs to tokenize both prompt and response)
            if self.use_response:
                assert response_message is not None, (
                    "response message is none but you want to tokenize prompt and response"
                )
                assert isinstance(response_message, dict)

                # RECAP prefill (SFT/eval path): render the counter-aligned CoT
                # as an *unfinished* assistant turn (continue_final_message=True)
                # so it becomes a prefix the model must continue from.
                if self.use_prefill and has_prefill_col:
                    assert isinstance(data["prefill"], str)
                    prefill_message = self.to_message(
                        role="assistant", content=data["prefill"]
                    )
                    # temporary use prefill as the response
                    conversation_message = prompt_message + [prefill_message]
                    self.chat_template_kwargs["add_generation_prompt"] = False
                    self.chat_template_kwargs["continue_final_message"] = True

                    conversation_text = self.tokenizer.apply_chat_template(
                        conversation_message,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )
                    # restore the true prompt+label conversation so downstream
                    # reward/metric scoring never sees the injected prefill.
                    conversation_message = prompt_message + [response_message]

                # no prefilling, sft case, so we need to compute the prompt_text len to mask out the label
                else:
                    conversation_message = prompt_message + [response_message]
                    prompt_template_kwargs = self.chat_template_kwargs.copy()
                    prompt_template_kwargs["add_generation_prompt"] = True
                    prompt_template_kwargs["continue_final_message"] = False
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_message,
                        tokenize=False,
                        **prompt_template_kwargs,
                    )
                    conversation_text = self.tokenizer.apply_chat_template(
                        conversation_message,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )

            # only tokenize prompt, so we dont care if the response message is none or not, but we will still save the lable response if provided, rlhf only tokenize the prompt (w/ or w/o prefill), but may need label for RM to judge
            else:
                # RECAP prefill (RLHF rollout path): same idea, but only the
                # prompt+prefill is tokenized as model input. prompt_ids_len
                # (computed below) will include the prefill, so the rollout
                # treats these injected tokens as fixed context, not actions.
                if self.use_prefill and has_prefill_col:
                    assert isinstance(data["prefill"], str)
                    prefill_message = self.to_message(
                        role="assistant", content=data["prefill"]
                    )
                    # temporary use prefill as the response
                    conversation_message = prompt_message + [prefill_message]
                    self.chat_template_kwargs["add_generation_prompt"] = False
                    self.chat_template_kwargs["continue_final_message"] = True

                    conversation_text = self.tokenizer.apply_chat_template(
                        conversation_message,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )
                    # change back to normal conversation for metric and eval
                    conversation_message = (
                        prompt_message
                        if response_message is None
                        else prompt_message + [response_message]
                    )

                # no prefilling
                else:
                    conversation_message = prompt_message
                    self.chat_template_kwargs["add_generation_prompt"] = True
                    self.chat_template_kwargs["continue_final_message"] = False
                    conversation_text = self.tokenizer.apply_chat_template(
                        conversation_message,
                        tokenize=False,
                        **self.chat_template_kwargs,
                    )
                    if response_message is not None:
                        conversation_message = prompt_message + [response_message]

        else:
            # prefilling is not supported without chat template
            assert isinstance(prompt_message, str)
            prompt_text = prompt_message
            if self.use_response:
                assert response_message is not None and isinstance(
                    response_message, str
                )
                conversation_text = f"{prompt_message} {response_message}"
                conversation_message = [conversation_text]
            else:
                conversation_text = prompt_message
                if response_message is not None:
                    conversation_message = [conversation_text, response_message]

        if self.pretrain_mode:
            prompt_ids_len = 0
        else:
            # prompt_ids_len marks the token boundary that separates fixed
            # context from trainable/label tokens. When prefilling, prompt_text
            # is None and conversation_text already contains prompt + injected
            # CoT prefix, so this length equals ℓ_pre (prompt + prefill) — every
            # token after it is what the policy generates and is optimized on.
            # we dont apply prompt_max_len constraint on sft data, we only apply total length constraint to sft data
            conversation_token = self.tokenizer(
                conversation_text if prompt_text is None else prompt_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                add_special_tokens=not self.apply_chat_template,
                return_length=True,
            )
            prompt_ids_len = conversation_token["length"][0]

        # save a copy of the conversation here so that in rlhf rollout experience, we can reuse the conversation
        return {
            "model_input": conversation_text,
            "conversation": conversation_message,  # the response here should be label, not prefill
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges,
            "value": value,
            "position": position,
            "reward": None if "reward" not in data else data["reward"],
            "hit_max_length": False if prompt_ids_len < self.max_length else True,
        }

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        model_input = self.model_inputs[idx]
        conversation = self.conversations[idx]
        reward = self.rewards[idx]

        model_input_token = self.tokenizer(
            model_input,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=not self.apply_chat_template,
            return_length=True,
        )

        # the input length is the length of prompt + response
        input_length = model_input_token["length"][0]
        info = {
            "model_input": model_input,
            "conversation": conversation,
            "input_length": input_length,
        }

        if reward is not None:
            info["reward"] = reward

        if self.multiturn:
            info["response_ranges"] = self.response_ranges[idx]

        if self.use_value:
            info["value"] = torch.tensor(self.values[idx])
            info["position"] = torch.tensor(self.positions[idx])
            # because we use split() and " ".join() in creating values, which may cause the sequence length different from the original one.
            info["position"][-1] = input_length

        return (prompt_ids_len, model_input_token, info)

    def collate_fn(
        self, item_list: Tuple, padding: bool = True, padding_side: str = "right"
    ):
        # performs a transpose, grouping elements by position across all tuples.
        prompt_ids_lens, input_tokens, infos = zip(*item_list)

        # if we use input_length in final_info, we need to pad on right to ensure the prompt_ids_len is correct or no padding
        # no padding for rlhf online sampling
        if padding:
            input_tokens = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                input_tokens,
                return_tensors="pt",
                padding_side=padding_side,
            )
        # no padding for rlhf online sampling, we will return a List[List[int]]
        else:
            input_tokens = [i["input_ids"] for i in input_tokens]

        info_model_input = [item["model_input"] for item in infos]
        info_conversation = [item["conversation"] for item in infos]
        info_input_length = torch.tensor([item["input_length"] for item in infos])

        final_info = {
            "input": info_model_input,
            "conversation": info_conversation,
            "input_length": info_input_length,
        }

        if "reward" in infos[0]:
            info_reward = [item["reward"] for item in infos]
            final_info["reward"] = info_reward

        # value sft training
        if self.use_value:
            info_valus = [item["value"] for item in infos]
            info_positions = [item["position"] for item in infos]

            info_valus = pad_sequence(
                info_valus,
                batch_first=True,
                padding_value=0.0,
                padding_side=padding_side,
            )  # shape: (batch_size, max_chunk_num)
            info_positions = pad_sequence(
                info_positions,
                batch_first=True,
                padding_value=-1,
                padding_side=padding_side,
            )

            final_info["value"] = info_valus
            final_info["position"] = info_positions

        return (
            prompt_ids_lens,
            input_tokens,
            final_info,
        )

    def packing_collate_fn(self, item_list: Tuple):
        raise NotImplementedError
