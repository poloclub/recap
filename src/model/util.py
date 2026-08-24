"""Shared model/tokenizer loading and generation helpers.

Used by the RLHF Ray actors (tokenizer setup) as well as by the deprecated
FSDP inference and loss-landscape CLIs (:mod:`src.deprecated_cli`) for
loading plain HuggingFace models/anchors and running batched generation.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonlines
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm
from omegaconf import DictConfig
from torch import nn
from torch.distributed import checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.data.util import batch_to_device

__all__ = [
    "load_model_and_tokenizer",
    "get_all_reward_models_from_data_cfg",
    "save_generation",
    "get_max_tp_size_from_list",
    "load_model_and_tokenizer_and_anchors",
    "load_sharded_weights",
    "get_tokenizer",
    "compute_tp_size_from_model_size",
]

logger = logging.getLogger(__name__)


def compute_tp_size_from_model_size() -> int:
    """Placeholder for auto-selecting a tensor-parallel size from model size; currently always returns 1."""
    return 1


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    """Load a tokenizer, defaulting the pad token to eos if unset."""
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_all_reward_models_from_data_cfg(
    data_cfg: DictConfig, reward_cfg: DictConfig
) -> List[str]:
    """Return the distinct reward-model names referenced by the dataset configs (``d0``, ``d1``, ...)."""
    # 1. validate all RM are available in existing RM cfg
    all_reward_models = list(reward_cfg.keys())

    # 2. extract RM from data_cfg.dx
    reward_models_requested_by_data = []
    for name, ds_cfg in data_cfg.items():
        # dataset naming convention in hydra: dx, where x is a number
        if not (name.startswith("d") and name[1:].isdigit()):
            continue
        reward_name = ds_cfg.reward_name
        if reward_name in all_reward_models:
            reward_models_requested_by_data.append(reward_name)
        else:
            raise ValueError(
                f"Requested reward model type {reward_name} not defined in {all_reward_models}"
            )

    return list(set(reward_models_requested_by_data))


def get_max_tp_size_from_list(counts: List[int], world_size: int) -> int:
    """
    Returns the largest tensor parallel size ≤ world_size that divides all counts.
    counts may include num_attention_heads, vocab size

    Args:
        head_counts (List[int]): List of attention head counts to be divisible.
        world_size (int): Maximum available GPU world size.

    Returns:
        int: Best TP size that divides all elements in head_counts and is ≤ world_size.
    """
    for tp_size in reversed(range(1, world_size + 1)):
        if all(h % tp_size == 0 for h in counts):
            return tp_size
    return 1  # Fallback if nothing else works


def load_model_and_tokenizer(
    model_name_or_path,
    config_args: Optional[dict] = None,
    model_args: Optional[dict] = None,
    tokenizer_args: Optional[dict] = None,
):
    """Load a plain HuggingFace causal LM and tokenizer (no DeepSpeed/vLLM)."""
    if config_args is None:
        config_args = {}
    if model_args is None:
        model_args = {}
    if tokenizer_args is None:
        tokenizer_args = {}
    config = AutoConfig.from_pretrained(model_name_or_path, **config_args)

    dtype = torch.float32
    if "torch_dtype" in model_args:
        dtype = model_args.torch_dtype
        del model_args.torch_dtype
        if dtype in ["float32", "fp32"]:
            dtype = torch.float32
        elif dtype in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        else:
            raise NotImplementedError
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=dtype, **model_args
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_model_and_tokenizer_and_anchors(
    model_name_or_path_list: List[str],
    config_args: Optional[dict] = None,
    model_args: Optional[dict] = None,
    tokenizer_args: Optional[dict] = None,
    use_grad: Optional[bool] = False,
):
    """load empty model and anchors for MetaLLM
    Each model will first be loaded to cpu, disable gradient, and be prepared in FSDP
    """
    if config_args is None:
        config_args = {}
    if model_args is None:
        model_args = {}
    if tokenizer_args is None:
        tokenizer_args = {}
    model_name_or_path = model_name_or_path_list[0]
    config = AutoConfig.from_pretrained(model_name_or_path, **config_args)

    dtype = torch.float32
    if "torch_dtype" in model_args:
        dtype = model_args.torch_dtype
        del model_args.torch_dtype
        if dtype in ["float32", "fp32"]:
            dtype = torch.float32
        elif dtype in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        else:
            raise NotImplementedError

    # non-empty model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=dtype, **model_args
    )
    if not use_grad:
        for param in model.parameters():
            param.requires_grad = False

    anchors = []
    for model_name_or_path in model_name_or_path_list:
        anchor = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=config, torch_dtype=dtype, **model_args
        )
        if not use_grad:
            for param in anchor.parameters():
                param.requires_grad = False
        anchors.append(anchor)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, anchors


def save_generation(
    output: list[dict],
    save_to: Union[Path, str] = None,
    filename: Optional[str] = None,
):
    """Write generation records to a per-device JSONL file under ``save_to``."""
    if not filename:
        filename = f"device{torch.cuda.current_device()}.jsonl"

    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        save_to_file = Path(save_to) / filename
    else:
        save_to_file = Path(filename)

    logger.info(f"Saving prompts and generations to {save_to_file.resolve()}")
    with jsonlines.open(save_to_file, "w") as f:
        f.write_all(output)


def forward_llm(
    model: nn.Module,
    accelerator: Accelerator,
    dataset: List[List],
    collate_fn: Callable,
    generation_args: Dict,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset_length: Optional[int] = None,
    use_fsdp: Optional[bool] = True,
):
    """Run batched generation over ``dataset`` split across accelerate processes, gathering the results.

    Used by the deprecated FSDP inference / loss-landscape CLIs, not by the
    main RLHF pipeline.
    """
    if dataset_length is None:
        dataset_length = sum([len(i) for i in dataset])

    outputs_per_process = []
    # assuming the dataset has been batched
    with accelerator.split_between_processes(
        dataset, apply_padding=True
    ) as batched_inputs:
        for batch in tqdm(batched_inputs, desc="Generating"):
            names = list(batch[0].keys())
            assert "input_ids" in names and "attention_mask" in names
            names.remove("input_ids")
            names.remove("attention_mask")
            model_inputs = [
                {k: v for k, v in entry.items() if k in ["input_ids", "attention_mask"]}
                for entry in batch
            ]
            model_inputs = collate_fn(model_inputs)
            model_inputs = batch_to_device(model_inputs, accelerator.device)

            if use_fsdp:
                with FSDP.summon_full_params(model, writeback=False):
                    outputs = model.module.generate(
                        **model_inputs,
                        **generation_args,
                        pad_token_id=tokenizer.pad_token_id,
                        synced_gpus=True,
                    )
            else:
                outputs = model.generate(
                    **model_inputs,
                    **generation_args,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_length = model_inputs["input_ids"].shape[-1]
            prompt_text = tokenizer.batch_decode(
                outputs[:, :input_length], skip_special_tokens=True
            )
            generated_text = tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )

            outputs_local = [
                {"prompt": i, "generation": j}
                for i, j in zip(prompt_text, generated_text)
            ]

            # add groundtruth or additional columns here
            if len(names) > 0:
                for entry, obj in zip(outputs_local, batch):
                    for name in names:
                        entry[name] = obj[name]

            outputs_per_process += outputs_local

    outputs_gather = gather_object(outputs_per_process)
    # Drop duplicates produced by apply_padding in split_between_processes
    final_output = outputs_gather[:dataset_length]

    return final_output


def load_sharded_weights(
    model: nn.Module, sharded_weights_path: Union[str, Path] = None
):
    """Load a distributed (sharded) checkpoint into ``model``'s state dict."""
    reader = checkpoint.FileSystemReader(sharded_weights_path)
    state_dict = {"model": model.state_dict()}
    checkpoint.load_state_dict(
        state_dict=state_dict, storage_reader=reader, no_dist=True
    )
    model = model.load_state_dict(state_dict["model"])
    return model
