"""Shared helper utilities used across the training/inference entry points.

Includes: interactive-chat conversation state management, resolving a model
name to a local checkpoint vs. a HuggingFace Hub id, sizing/validating Ray
placement groups for online RLHF from the Hydra config, picking a tensor
parallel size compatible with a model's architecture, and small process/IO
utilities (seeding, temp-directory cleanup).
"""

import logging
import shutil
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Union

import numpy as np
import ray
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
)
from transformers import AutoConfig

from src.data import detect_message_type, load_system_prompt_from_file
from src.model import (
    compute_tp_size_from_model_size,
    get_all_reward_models_from_data_cfg,
    get_max_tp_size_from_list,
)

logger = logging.getLogger(__name__)


# clear the conversation in interactive chat
def reset_chat(
    apply_chat_template: bool = True,
    system: str = None,
    chat_template: str = None,
) -> Union[List[Dict[str, str]], str]:
    """Build a fresh conversation state for interactive chat.

    Loads the optional system prompt (``system`` is either a literal string
    or a path to a ``.txt`` file) and returns either a list of chat messages
    (when ``apply_chat_template`` is True, using the message format implied
    by ``chat_template``) or a plain text prefix.
    """
    system = load_system_prompt_from_file(system)

    if apply_chat_template:
        assert chat_template is not None
        to_message = detect_message_type(chat_template)
        if system is not None:
            user_prompt = [to_message(role="system", content=system)]
        else:
            user_prompt = []
    else:
        user_prompt = f"{system}\n" if system is not None else ""

    return user_prompt


def get_model_path(output_dir: Path, model_name: str) -> str:
    """Resolve ``model_name`` to a local checkpoint path under ``output_dir``
    if one exists there, otherwise fall back to treating it as a
    HuggingFace Hub id.
    """
    # ray will hijack the hydra working directory, and does not recognize pathlib directory
    model_path = (output_dir / model_name).resolve()
    if model_path.exists():
        model_path = str(model_path)  # from local
        logger.info(f"Loading local model from {model_path} ...")
    else:
        model_path = model_name  # from hf
        logger.info(f"Loading HF model from {model_path} ...")
    return model_path


def setup_ray_pg_from_hydra_config(cfg: DictConfig, ray_resources: Dict):
    """Validate the online RLHF GPU/batch-size configuration in ``cfg.train``
    and build the Ray placement group(s) needed to launch it.

    Checks that the GPU counts requested for the policy, reference, critic,
    and (forward/generative) reward models fit within the Ray cluster and
    that rollout/training batch sizes divide evenly across each role's GPU
    allocation. If ``cfg.train.colocate_all_models`` or
    ``colocate_actor_ref`` is set, creates a single ``PACK`` placement group
    shared by the colocated roles.

    Returns:
        Tuple of ``(pg_rollout, pg_policy, pg_ref, reward_model_gpu_cfg)``,
        where each ``pg_*`` is a Ray placement group (or ``None`` if that
        role isn't colocated) and ``reward_model_gpu_cfg`` maps reward model
        name to its GPU/engine allocation.
    """
    # verify total requested gpus
    total_gpus = ray_resources["GPU"]
    policy_num_gpus = cfg.train.policy_num_gpus
    ref_num_gpus = cfg.train.ref_num_gpus
    critic_num_gpus = cfg.train.critic_num_gpus
    forward_reward_num_gpus = cfg.train.forward_reward_num_gpus
    generative_reward_num_gpus = cfg.train.generative_reward_num_gpus
    assert (
        policy_num_gpus
        + ref_num_gpus
        + critic_num_gpus
        + forward_reward_num_gpus
        + generative_reward_num_gpus
        <= total_gpus
    ), (
        f"[GPUS]: policy: {policy_num_gpus}, reference: {ref_num_gpus}, critic: {critic_num_gpus}, forward RM: {forward_reward_num_gpus}, generative RM: {generative_reward_num_gpus} shouldn't be greater than {total_gpus} available GPUs."
    )
    total_rollout_prompts = (
        cfg.train.vllm_generate_batch_size * cfg.train.n_samples_per_prompt
    )

    # rollout batch size check - deepspeed zero 3 hangs if non-divisble micro batch size
    mbs = cfg.train.micro_rollout_batch_size
    # ==== policy
    policy_rollout_batch_size = policy_num_gpus * mbs
    assert total_rollout_prompts % policy_rollout_batch_size == 0, (
        f"Invalid rollout: {total_rollout_prompts} rollout samples must be divisible by and greater than {policy_rollout_batch_size} policies"
    )

    # ==== reference
    ref_rollout_batch_size = ref_num_gpus * mbs
    if cfg.train.init_kl_coef > 0:
        assert ref_num_gpus > 0, "No reference gpus scheduled for computing KL"
        assert total_rollout_prompts % ref_rollout_batch_size == 0, (
            f"Invalid rollout: {total_rollout_prompts} rollout samples must be divisible by and greater than {ref_rollout_batch_size} references"
        )

    # ==== critic
    critic_rollout_batch_size = critic_num_gpus * mbs
    if critic_rollout_batch_size > 0:
        assert total_rollout_prompts % critic_rollout_batch_size == 0, (
            f"Invalid rollout: {total_rollout_prompts} rollout samples must be divisible by and greater than {critic_rollout_batch_size} critics"
        )

    # ==== reward
    reward_model_name_list = get_all_reward_models_from_data_cfg(
        data_cfg=cfg.data, reward_cfg=cfg.reward_model
    )
    multi_rm_cfg = cfg.reward_model
    forward_rm = []
    generative_rm = []
    reward_model_gpu_cfg = {}

    # first pass to get total generative and forward reward count
    for name in reward_model_name_list:
        single_rm_cfg = multi_rm_cfg[name]
        assert single_rm_cfg.type in ["rule", "forward", "generative"], (
            f"reward type {single_rm_cfg.type} not supported"
        )
        if single_rm_cfg.type == "forward":
            forward_rm.append(name)
        elif single_rm_cfg.type == "generative":
            generative_rm.append(name)
    forward_rm = list(set(forward_rm))
    generative_rm = list(set(generative_rm))
    assert forward_reward_num_gpus >= len(forward_rm), (
        "No sufficient GPUs for forward RM"
    )
    assert generative_reward_num_gpus >= len(generative_rm), (
        "No sufficient GPUs for generative RM"
    )

    # second pass to allocate gpu counts
    if forward_rm:
        gpus_per_forward_rm = forward_reward_num_gpus // len(forward_rm)
        forward_rm_rollout_batch_size = gpus_per_forward_rm * mbs
        assert total_rollout_prompts % forward_rm_rollout_batch_size == 0, (
            f"Invalid rollout: {total_rollout_prompts} rollout samples must be divisible by and greater than {forward_rm_rollout_batch_size} forward RMs"
        )
        for name in forward_rm:
            reward_model_gpu_cfg[name] = {"world_size": gpus_per_forward_rm}

    if generative_rm:
        gpus_per_generative_rm = generative_reward_num_gpus // len(generative_rm)
        for name in generative_rm:
            tp_size = compute_tp_size_from_model_size()
            assert tp_size <= gpus_per_generative_rm, (
                f"increase gpus for {name} generative RM, current: {gpus_per_generative_rm}, minimum {tp_size}"
            )
            reward_model_gpu_cfg[name] = {
                "vllm_num_engines": gpus_per_generative_rm // tp_size,
                "vllm_tensor_parallel_size": tp_size,
            }

    # ==== policy training
    policy_train_batch_size = (
        cfg.train.micro_train_batch_size * cfg.train.policy_num_gpus
    )
    assert policy_train_batch_size <= total_rollout_prompts, (
        f"Total rollout samples {total_rollout_prompts} should be greater than {policy_train_batch_size} policy training batch size"
    )
    assert policy_train_batch_size <= cfg.train.train_batch_size, (
        f"policy training batch size mismatch: provided {cfg.train.train_batch_size}, but needs at least {policy_train_batch_size}, thus grad accum will throw error"
    )
    if total_rollout_prompts % policy_train_batch_size != 0:
        logger.warning(
            f"{total_rollout_prompts} rollout prompts not divisible by policy training batch size {policy_train_batch_size}; some will be dropped by the DataLoader."
        )

    # ==== create placement group
    # colocate all models & colocate actor reference
    if cfg.train.colocate_all_models or cfg.train.colocate_actor_ref:
        if cfg.train.init_kl_coef > 0:
            assert cfg.train.policy_num_gpus == cfg.train.ref_num_gpus, (
                "num_gpus must be the same when colocate actor and ref model."
            )

        world_size = cfg.train.policy_num_gpus
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

        if cfg.train.colocate_all_models:
            pg_rollout = pg_policy = pg_ref = pg
        elif cfg.train.colocate_actor_ref:
            pg_policy = pg_ref = pg
            pg_rollout = None
    else:
        pg_rollout = pg_policy = pg_ref = None

    # for vllm rollout engines
    if cfg.train.colocate_all_models and not cfg.train.async_train:
        assert (
            cfg.train.policy_num_gpus
            == cfg.train.vllm_num_engines * cfg.train.vllm_tensor_parallel_size
        ), (
            f"policy_num_gpus must be equal to vllm_num_engines * vllm_tensor_parallel_size, got {policy_num_gpus} and {cfg.train.vllm_num_engines * cfg.train.vllm_tensor_parallel_size}"
        )
        assert pg_rollout is not None

    logger.info(
        f"GPU settings: Rollout: {cfg.train.vllm_num_engines} engines * tp_size {cfg.train.vllm_tensor_parallel_size}, Policy: {policy_num_gpus} GPUs, Reference: {ref_num_gpus} GPUs, Critic: {critic_num_gpus} GPUs\nReward Models: {reward_model_gpu_cfg}"
    )

    if cfg.train.colocate_actor_ref or cfg.train.colocate_all_models:
        logger.info(
            f"Creating PG when colocating modles: {pformat(placement_group_table(pg))}"
        )

    return pg_rollout, pg_policy, pg_ref, reward_model_gpu_cfg


def determine_max_tensor_parallel(model_name: str):
    """determine the max tp size for a given model config, needs to be smaller than world size, divisible by several kwargs"""
    # Total number of attention heads must be divisible by tensor parallel size.
    max_tp_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name)
    # Uses a compact list comprehension with the walrus operator (:=) to assign and check in one line.
    counts = [
        val
        for attr in ["num_attention_heads", "vocab_size"]
        if isinstance(val := getattr(config, attr, None), int)
    ]

    if counts:
        max_tp_size = get_max_tp_size_from_list(counts, world_size=max_tp_size)
    logger.info(f"Using GPU world size: {max_tp_size}")

    return max_tp_size


def ensure_empty_parent_dir(file_path: Union[str, Path]):
    """Ensure ``file_path``'s parent directory exists and is empty, deleting
    any existing files/subdirectories inside it."""
    file_path = Path(file_path)
    parent_dir = file_path.parent

    # ensure parent dir exists
    parent_dir.mkdir(parents=True, exist_ok=True)

    # remove all contents in parent dir
    for item in parent_dir.iterdir():
        if item.is_file():
            item.unlink()  # remove file
        elif item.is_dir():
            shutil.rmtree(item)  # Remove directory and its contents


def set_seed(seed: int = 42):
    """Seed NumPy and PyTorch (CPU/CUDA) RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dist.is_initialized():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
