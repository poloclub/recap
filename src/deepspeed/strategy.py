"""Wraps a model in a DeepSpeed inference engine (ZeRO-inference config plus
optional tensor parallelism) for fast generation, as opposed to the DeepSpeed
ZeRO training strategy configured elsewhere via OpenRLHF's DeepspeedStrategy.
"""

import torch.distributed as dist
from omegaconf import DictConfig
from openrlhf.models import Actor
from transformers import PreTrainedModel

import deepspeed

__all__ = ["setup_deepspeed_for_inference"]


def setup_deepspeed_for_inference(
    model: PreTrainedModel,
    ds_cfg: DictConfig,
    max_length: int,
):
    """Build a `deepspeed.init_inference` engine for `model` and swap it in.

    Args:
        model: an OpenRLHF `Actor` (HF model wrapped in `.model`) or a plain
            HF `PreTrainedModel`.
        ds_cfg: config with `zero_stage`; tensor parallelism across the full
            world size is only enabled when `zero_stage == 3`.
        max_length: max output token length for the inference engine.

    Returns:
        The `Actor` with its inner model replaced by the inference engine, or
        the inference engine itself if `model` was not an `Actor`.
    """
    is_actor = isinstance(model, Actor)

    stage = ds_cfg.zero_stage

    # Define Tensor Parallelism Config: https://deepspeed.readthedocs.io/en/stable/inference-init.html
    tp_config = {
        "enabled": True,
        "tp_size": dist.get_world_size() if stage == 3 else 1,  # Tensor Parallelism
    }
    zero_config = {"stage": stage}
    config = {
        "tensor_parallel": tp_config,
        "max_out_tokens": max_length,
        "zero": zero_config,
    }

    engine = deepspeed.init_inference(
        model=model.model if is_actor else model, config=config
    )

    if is_actor:
        model.model = engine
    else:
        model = engine

    return model
