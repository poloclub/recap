"""Online RLHF entry point (GRPO / DAPO), and the launcher for RECAP training.

Run via ``python -m src.rlhf`` (typically through ``make rlhf name=...``), which
loads ``config/rlhf.yaml``. This module is the central controller: it validates
the Ray/GPU layout, spins up the vLLM rollout engines and the Ray actor groups
for the policy, (optional) reference, critic, and one-or-more reward models,
then hands them to :class:`~src.trainer.RLHFTrainer` and runs the
rollout → reward → policy-update loop.

RECAP requires no changes here: counter-aligned prefills are supplied through the
dataset (``data.use_prefill``; see :mod:`src.data.dataset`) and the prefix is
excluded from optimization inside the trainer, so the same GRPO/DAPO machinery
trains both standard and prefilled prompts.
"""

import logging
import os
from pathlib import Path
from pprint import pformat

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from openrlhf.utils.deepspeed import DeepspeedStrategy

from src.model import (
    GenerativeRewardModelActorGroup,
    LLMRayActor,
    PolicyModelActor,
    RayActorGroup,
    ReferenceModelActor,
    RuleRewardModelActor,
    create_vllm_engines,
    get_all_reward_models_from_data_cfg,
)
from src.trainer import RLHFTrainer

from .misc import get_model_path, setup_ray_pg_from_hydra_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="rlhf", version_base="1.3")
def main(cfg: DictConfig):
    """Assemble the distributed RLHF job from the Hydra config and run it.

    Steps: resolve output/model paths → init Ray → plan placement groups →
    build the DeepSpeed strategy → create vLLM engines → construct the policy /
    reference / (skipped) critic / reward actor groups → initialize all models →
    ``RLHFTrainer.fit()`` → save the final policy in HF format.
    """
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    # ==== checkpoint and final HF model path
    cfg.log.ckpt_path = str(output_dir / cfg.log.ckpt_path)
    cfg.log.save_path = str(output_dir / cfg.log.save_path)
    cfg.model.policy_model.name = get_model_path(
        output_dir=output_dir, model_name=cfg.model.policy_model.name
    )
    cfg.model.ref_model.name = get_model_path(
        output_dir=output_dir, model_name=cfg.model.ref_model.name
    )
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # ==== initialize ray
    if not ray.is_initialized():
        logger.info("Initializing ray ...")
        ray.init(
            address="127.0.0.1:9988",
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            },
        )
    ray_resources = ray.cluster_resources()
    logger.info(f"Ray resources:\n{pformat(ray_resources)}")

    # ==== validate configs
    logger.info("Verifying resources and hardware configs ...")
    pg_rollout, pg_policy, pg_ref, reward_model_gpu_cfg = (
        setup_ray_pg_from_hydra_config(cfg, ray_resources)
    )

    # ==== deepspeed strategy
    strategy = DeepspeedStrategy(
        seed=cfg.seed,
        full_determinism=cfg.deepspeed.full_determinism,
        max_norm=cfg.train.max_norm,
        micro_train_batch_size=cfg.train.micro_train_batch_size,
        train_batch_size=cfg.train.train_batch_size,
        zero_stage=cfg.deepspeed.zero_stage,
        bf16=cfg.model.policy_model.bf16,
        args=cfg.deepspeed,
    )

    # ========== init vLLM engine for text generation
    vllm_engines = None
    if cfg.train.vllm_num_engines is not None and cfg.train.vllm_num_engines > 0:
        max_len = cfg.train.prompt_max_len + cfg.train.generate_max_len

        logger.info("Creating vllm engines ...")
        vllm_engines = create_vllm_engines(
            num_engines=cfg.train.vllm_num_engines,
            tensor_parallel_size=cfg.train.vllm_tensor_parallel_size,
            pretrain=cfg.model.policy_model.name,
            seed=cfg.seed,
            full_determinism=cfg.deepspeed.full_determinism,
            enable_prefix_caching=cfg.train.enable_prefix_caching,
            enforce_eager=cfg.train.enforce_eager,
            max_model_len=max_len,
            shared_pg=pg_rollout,
            gpu_memory_utilization=cfg.train.vllm_gpu_memory_utilization,
            vllm_enable_sleep=cfg.train.vllm_enable_sleep,
            llm_actor_cls=LLMRayActor,
        )

    # ========== policy model
    policy_model = RayActorGroup(
        num_gpus=cfg.train.policy_num_gpus,
        ray_actor_type=PolicyModelActor,
        pg=pg_policy,
        num_cpus_per_actor=0.2 if pg_policy else 1,
        num_gpus_per_actor=0.2 if pg_policy else 1,
        duplicate_actors=cfg.model.policy_model.ring_attn_size
        * cfg.deepspeed.ds_tensor_parallel_size,
    )

    # ========== reference model for KL
    if cfg.train.init_kl_coef <= 0:
        ref_model = None
    else:
        ref_model = RayActorGroup(
            num_gpus=cfg.train.ref_num_gpus,
            ray_actor_type=ReferenceModelActor,
            pg=pg_ref,
            num_cpus_per_actor=0.2 if pg_ref else 1,
            num_gpus_per_actor=0.2 if pg_ref else 1,
            duplicate_actors=cfg.model.ref_model.ring_attn_size
            * cfg.deepspeed.ds_tensor_parallel_size,
        )

    critic_model = None

    # ========== Reward model (rule, forward, generative)
    if cfg.train.remote_rm_url is not None:
        multi_reward_model = None
        raise NotImplementedError
    else:
        reward_model_name_list = get_all_reward_models_from_data_cfg(
            data_cfg=cfg.data, reward_cfg=cfg.reward_model
        )
        logger.info(f"All reward models used in training: {reward_model_name_list}")
        multi_rm_cfg = cfg.reward_model

        # additional kwargs for RM
        rm_kwargs = {}
        if cfg.model.policy_model.cot_end_token is not None:
            rm_kwargs["cot_end_token"] = cfg.model.policy_model.cot_end_token
        if cfg.train.use_overlong_filtering:
            rm_kwargs["use_overlong_filtering"] = True

        multi_reward_model = {}
        for name in reward_model_name_list:
            single_rm_cfg = multi_rm_cfg[name]
            assert single_rm_cfg.type in ["rule", "forward", "generative"], (
                f"reward type {single_rm_cfg.type} not supported"
            )
            logger.info(f"Loading {single_rm_cfg.type} reward model: {name} ...")
            # rule-based verifiable (no gpu)
            if single_rm_cfg.type == "rule":
                reward_model = RayActorGroup(
                    num_gpus=0,
                    ray_actor_type=RuleRewardModelActor,
                    pg=None,
                    num_cpus_per_actor=1,
                    num_gpus_per_actor=0,
                    duplicate_actors=1,
                )
                reward_model.rm_type = "rule"

            # forward RM that needs deepspeed
            elif single_rm_cfg.type == "forward":
                raise NotImplementedError
            # generative RM that needs vllm
            elif single_rm_cfg.type == "generative":
                reward_model = GenerativeRewardModelActorGroup(
                    vllm_num_engines=reward_model_gpu_cfg[name]["vllm_num_engines"],
                    vllm_tensor_parallel_size=reward_model_gpu_cfg[name][
                        "vllm_tensor_parallel_size"
                    ],
                    train_cfg=cfg.train,
                    model_cfg=single_rm_cfg,
                    seed=cfg.seed,
                    full_determinism=cfg.deepspeed.full_determinism,
                    **rm_kwargs,
                )
            else:
                raise ValueError("RM type not supported")

            # we need to store the cfg for that specific rm for future initialization
            multi_reward_model[name] = [reward_model, single_rm_cfg]

    # RLHF trainer (central controller)
    rlhf_trainer = RLHFTrainer(
        strategy=strategy,
        policy_model_group=policy_model,
        critic_model_group=critic_model,
        multi_reward_model_group=multi_reward_model,
        ref_model_group=ref_model,
        vllm_engines=vllm_engines,
        cfg=cfg,
    )
    max_steps = rlhf_trainer.get_max_steps()  # max training steps -> lr scheduler

    # initialize policy/reference/critic/reward model
    refs = policy_model.async_init_model_from_pretrained(
        strategy=strategy,
        vllm_engines=vllm_engines,
        max_steps=max_steps,
        model_cfg=cfg.model.policy_model,
        train_cfg=cfg.train,
        log_cfg=cfg.log,
    )
    if ref_model is not None:
        refs.extend(
            ref_model.async_init_model_from_pretrained(
                strategy=strategy, model_cfg=cfg.model.ref_model, train_cfg=cfg.train
            )
        )
    if critic_model is not None:
        raise NotImplementedError

    for rm_name, rm_sample in multi_reward_model.items():
        reward_model, reward_model_cfg = rm_sample
        if reward_model.rm_type != "generative":
            refs.extend(
                reward_model.async_init_model_from_pretrained(
                    strategy=strategy, model_cfg=reward_model_cfg, **rm_kwargs
                )
            )

    logger.info("Initializing all policy/reference/critic/reward models")
    ray.get(refs)

    # rollout -> training -> rollout -> ...
    logger.info("Start rollout -> policy training")
    rlhf_trainer.fit()

    # save policy model
    logger.info("Saving final policy model in HF format")
    ray.get(policy_model.async_save_model())


if __name__ == "__main__":
    main()
