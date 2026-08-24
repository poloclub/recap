"""Dataset helpers: column normalization, blending, prompt/response formatting.

Includes chat-template message construction, mapping each dataset's arbitrary
column names onto the canonical schema (``prompt`` / ``response`` / ``prefill`` /
...), interleaving multiple training sources by probability, and building
inference dataloaders. The ``prefill`` mapping is what lets RECAP select which
precomputed counter-aligned CoT column (e.g. ``pre500``, ``r050``) to inject.
"""

import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import datasets
import torch
import torch.distributed as dist
from datasets import (
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

__all__ = [
    "can_print",
    "detect_message_type",
    "format_prompt_and_response",
    "format_reward_pairs",
    "batch_to_device",
    "blend_datasets",
    "setup_dataloader_for_inference",
    "load_system_prompt_from_file",
    "zero_pad_sequences",
]

logger = logging.getLogger(__name__)


def can_print():
    return not dist.is_initialized() or dist.get_rank() == 0


def detect_message_type(chat_template: str) -> Callable:
    """Determine which type of dict to use in conversations
    multimodal:
    [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    standard:
    [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    """
    assert chat_template is not None

    # llamaguard 3-1b and multimodal style
    if '"type": "text"' in chat_template or "selectattr('type'" in chat_template:

        def to_message(role: str, content: List[str], type: str = "text"):
            if isinstance(content, list):
                return {
                    "role": role,
                    "content": [{"type": type, type: c} for c in content],
                }
            elif isinstance(content, str):
                return {
                    "role": role,
                    "content": [{"type": type, type: content}],
                }

    # standard
    else:

        def to_message(role: str, content: str) -> Dict:
            return {"role": role, "content": content}

    return to_message


def format_prompt_and_response(
    sample: Union[Dict[str, List], Dict[str, str]],
    apply_chat_template: bool = True,
    chat_template: str = None,
    for_guardrail: bool = False,
) -> Union[Tuple[List[Dict[str, str]], Dict[str, str]], Tuple[str, str]]:
    """
    Convert different input format to conversation format if apply chat template
    note, no tokenization happens in this step
    we assume a conversation format is needed when applying prompt template, otherwise we only return the prompt
    use chat template whenever possible

    merge system message into prompt if any, output prompt and response. the target of training will be prompt + response

    chat_template is only used when apply_chat_template is set to true, used to determine message type
    guardrail model typically doesn't need system prompt, we assume guardrail model always apply chat template

    """
    assert "prompt" in sample

    if apply_chat_template:
        assert chat_template is not None
        to_message = detect_message_type(chat_template)

    if "system" in sample:
        assert isinstance(sample["system"], str)

    # this is not conversation format and typically for single turn conversation
    if isinstance(sample["prompt"], str):
        if apply_chat_template:
            prompt = []
            if "system" in sample and not for_guardrail:
                system = to_message(role="system", content=sample["system"])
                prompt.append(system)
            prompt.append(to_message(role="user", content=sample["prompt"]))
            # if prompt is str, we assume the response should exist in a different column
            if "response" in sample:
                response = to_message(role="assistant", content=sample["response"])
            else:
                response = None
        else:
            if "system" in sample:
                prompt = " ".join([sample["system"], sample["prompt"]])
            else:
                prompt = sample["prompt"]
            response = sample["response"] if "response" in sample else None
    # this may/may not be conversation format, depends on whether the chat template is already applied or not
    # people may pass multiturn conversation as a list of str
    elif isinstance(sample["prompt"], list) and isinstance(sample["prompt"][0], str):
        if apply_chat_template:
            raise NotImplementedError
        else:
            prompt = " ".join(sample["prompt"])
            if "system" in sample:
                prompt = " ".join([sample["system"], prompt])

            response = None
            if "response" in sample:
                response = sample["response"]
                if isinstance(response, list):
                    response = " ".join(sample["response"])
                assert isinstance(response, str), (
                    f"response should be str, but got {type(response)}"
                )
    # we assume conversation format has been correctly applied
    elif isinstance(sample["prompt"], list) and isinstance(sample["prompt"][0], dict):
        if apply_chat_template:
            prompt = sample["prompt"]
            if "system" in sample and not for_guardrail:
                system = to_message(role="system", content=sample["system"])
                prompt = prompt.insert(0, system)
            if prompt[-1]["role"] == "assistant":
                response = prompt.pop()
            else:
                response = None
            if "response" in sample:
                if isinstance(sample["response"], dict):
                    assert sample["response"]["role"] == "assistant"
                    response = sample["response"]
                elif isinstance(sample["response"], str):
                    response = to_message(role="assistant", content=sample["response"])
                else:
                    raise ValueError("response type not accepted")
        else:
            prompt = sample["prompt"]
            # response may/may not be part of the input prompt
            if "response" in sample:
                response = sample["response"]
            else:
                if prompt[-1]["role"] == "assistant":
                    response = prompt.pop()["content"]
                else:
                    response = None
            prompt = " ".join([turn["content"] for turn in prompt])
            if "system" in sample:
                assert isinstance(sample["system"], str)
                prompt = " ".join([sample["system"], prompt])
    else:
        raise ValueError("data format is not accepted")

    return prompt, response


def format_reward_pairs(
    sample: Union[Dict[str, List], Dict[str, str]], apply_chat_template: bool = True
) -> Union[Tuple[List[Dict[str, str]], Dict[str, str]], Tuple[str, str]]:
    assert "chosen" in sample and "rejected" in sample
    assert isinstance(sample["chosen"], type(sample["rejected"]))

    system_in_prompt = "system" in sample
    prompt_in_chosen_rejected = "prompt" not in sample

    # already in the conversation format, e.g., openrlhf_mixture2
    if isinstance(sample["chosen"], list) and isinstance(sample["chosen"][0], dict):
        if apply_chat_template:
            # assume the final dict is the assistant response
            if prompt_in_chosen_rejected:
                prompt = sample["chosen"][:-1]
                chosen = sample["chosen"][-1]
                rejected = sample["rejected"][-1]
            else:
                prompt = sample["prompt"]
                chosen = sample["chosen"]
                rejected = sample["rejected"]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # margin loss (for RM training)
    margin = sample["margin"] if "margin" in sample else 0
    return prompt, chosen, rejected, margin


def batch_to_device(batch: dict, device: torch.device) -> dict:
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


def rename_dataset_columns(
    dataset: datasets.Dataset, ds_cfg: DictConfig
) -> datasets.Dataset:
    """single dataset only has one column for response, while pair dataset can have multi"""
    dataset_type = ds_cfg.type
    if dataset_type == "single":
        # we only allow three column names in the dataset: system, prompt, response
        column_name_mapping = {
            k: v
            for k, v in {
                ds_cfg.system_column: "system",
                ds_cfg.prompt_column: "prompt",
                # RECAP: whichever precomputed counter-aligned CoT column is
                # configured (e.g. pre100/pre500/r050) is renamed to "prefill",
                # which HydraConfigDataset then injects when use_prefill=True.
                ds_cfg.prefill_column: "prefill",
                ds_cfg.response_column: "response",
                ds_cfg.value_column: "value",
                ds_cfg.position_column: "position",
            }.items()
            if k is not None
        }
    elif dataset_type == "pair":
        column_name_mapping = {
            k: v
            for k, v in {
                ds_cfg.system_column: "system",
                ds_cfg.prompt_column: "prompt",
                ds_cfg.chosen_column: "chosen",
                ds_cfg.rejected_column: "rejected",
            }.items()
            if k is not None
        }
    else:
        raise NotImplementedError

    selected_column_names = list(column_name_mapping.keys())
    data = dataset.select_columns(selected_column_names).rename_columns(
        column_name_mapping
    )

    return data


def blend_datasets(datasets_cfg: DictConfig, seed: int):
    train_data_list = []
    eval_data_list = []
    prob_list = []

    use_eval = datasets_cfg.use_eval
    stopping_strategy = datasets_cfg.stopping_strategy
    order = datasets_cfg.order

    # we only support two types of dataset: 1) HF dataset and 2) stored locally as json, jsonl, or csv
    for name, ds_cfg in datasets_cfg.items():
        # dataset naming convention in hydra: dx, where x is a number
        if not (name.startswith("d") and name[1:].isdigit()):
            continue

        if can_print():
            logger.info(f"Loading {ds_cfg.id} dataset ...")

        data_files = ds_cfg.data_files
        if data_files is not None:
            data_files = OmegaConf.to_container(ds_cfg.data_files, resolve=True)
        data = load_dataset(
            path=ds_cfg.path,
            name=ds_cfg.name,
            data_files=data_files,
        )

        prob_list.append(ds_cfg.probability)

        # rename the columns based on the dataset type
        data = rename_dataset_columns(data, ds_cfg)

        # add reward type for rlhf dataset
        reward_name = ds_cfg.reward_name
        if reward_name is not None:

            def inject_reward_name_column(sample):
                sample["reward"] = reward_name
                return sample

            data = data.map(inject_reward_name_column)

        assert ds_cfg.train_split in data, (
            f"train_split {ds_cfg.train_split} not in {ds_cfg.id} dataset"
        )
        train_data = data[ds_cfg.train_split]
        max_samples = ds_cfg.max_samples
        if max_samples:
            max_samples = int(max_samples)
            max_idx = min(max_samples, len(data[ds_cfg.train_split]))
            train_data = train_data.filter(
                lambda _, idx: idx < max_idx, with_indices=True
            )

        train_data_list.append(train_data)

        if use_eval:
            eval_max_samples = 50
            if ds_cfg.eval_split in data:
                eval_data = data[ds_cfg.eval_split].select(
                    range(min(eval_max_samples, len(data[ds_cfg.eval_split])))
                )
            # if no eval split, but we still need to return eval, we will select 3% from training, and ensure there is at least one sample
            else:
                eval_data = train_data.select(
                    range(max(min(eval_max_samples, int(len(train_data) * 0.1)), 1))
                )
            eval_data_list.append(eval_data)

    if None in prob_list:
        prob_list = None
    else:
        assert math.isclose(sum(prob_list), 1.0)

    assert order in ["interleave", "concatenate"]
    if order == "interleave":
        train_dataset = interleave_datasets(
            train_data_list,
            probabilities=prob_list,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        # concatenate all datasets in order
        train_dataset = concatenate_datasets(train_data_list)

    if use_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=prob_list,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        eval_dataset = None

    if can_print():
        logger.info(
            f"Training dataset:\n{train_data_list}\nEvaluation dataset:\n{eval_data_list}\nAfter blending: {len(train_dataset)} training samples, {len(eval_dataset) if use_eval else 0} evaluation samples"
        )

    return train_dataset, eval_dataset


# adapted from https://github.com/OpenRLHF/OpenRLHF/blob/796f6f2529eb5ce1bc1dfd4defacf936cb93ceb2/openrlhf/utils/deepspeed/deepspeed.py#L147
def setup_dataloader_for_inference(
    replay_buffer,
    batch_size: int,
    pin_memory: bool = True,
    shuffle: bool = False,
    collate_fn=None,
    drop_last=False,
    seed: int = 42,
):
    # we need to set shuffle to false so that the distribute sampler will only repeat the samples from the beginning of the dataset
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    sampler = DistributedSampler(
        replay_buffer,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        replay_buffer,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return dataloader


def load_system_prompt_from_file(system: str = None):
    # system prompt can be passed from a txt or directly as a string
    system_prompt = None

    if system is not None:
        if system.endswith(".txt"):
            if Path(system).is_file():
                with open(system, "r") as f:
                    system_prompt = f.read()
        else:
            system_prompt = str(system)

    if can_print():
        logger.info(
            f"Loading system prompt from: {system}\nThe system prompt is: {system_prompt}"
        )

    return system_prompt

def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)
