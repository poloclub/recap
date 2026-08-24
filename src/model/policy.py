"""Ray actor for the trainable policy (actor) model in RLHF/GRPO/DAPO training.

Wraps the HuggingFace policy model behind a Ray remote actor: loads it under
a DeepSpeed strategy, computes action log-probs for rollout sequences, and
delegates optimization to :class:`~src.model.policy_trainer.PolicyTrainer`
using experiences collected in its replay buffer. Also broadcasts updated
weights to the vLLM rollout engines and manages checkpointing. Rollout
sequences may contain a fixed counter-aligned CoT prefix (RECAP); the
associated action mask (built during rollout) excludes that prefix so
training only optimizes the tokens generated after it.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import ray
import torch
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from openrlhf.utils.distributed_util import (
    torch_dist_barrier_and_cuda_sync,
)
from ray.actor import ActorHandle
from transformers import get_scheduler

from src.data import Experience

from .policy_trainer import PolicyTrainer
from .ray_actor_group import BaseModelActor
from .util import get_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["PolicyModelActor"]


# actor model
@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    """Ray actor that owns the trainable policy model and its DeepSpeed-backed trainer."""

    def init_model_from_pretrained(
        self,
        strategy: DeepspeedStrategy,
        vllm_engines: List[ActorHandle],
        max_steps: int,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        log_cfg: DictConfig,
    ):
        """Build the policy model, optimizer, scheduler, and PolicyTrainer; optionally restore a checkpoint."""
        self.vllm_engines = vllm_engines
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.max_steps = max_steps

        if train_cfg.vllm_num_engines > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if train_cfg.vllm_sync_backend == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        policy = Actor(
            pretrain_or_model=model_cfg.name,
            use_flash_attention_2=model_cfg.flash_attn,
            bf16=model_cfg.bf16,
            load_in_4bit=model_cfg.load_in_4bit,
            lora_rank=model_cfg.lora.rank,
            lora_alpha=model_cfg.lora.alpha,
            target_modules=model_cfg.lora.target_modules,
            lora_dropout=model_cfg.lora.dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=train_cfg.packing_samples,
            temperature=model_cfg.generation_args.temperature,
            use_liger_kernel=train_cfg.use_liger_kernel,
        )

        if strategy.is_rank_0():
            logger.info(f"Policy model:\n{policy}")

        # the tokenizer is only used while saving models, so padding side doesn't matter
        self.tokenizer = get_tokenizer(
            pretrain=model_cfg.name,
            model=policy.model,
            padding_side="left",
            use_fast=not model_cfg.disable_fast_tokenizer,
        )
        # overwrite incorrect deepseek model chat template
        if model_cfg.chat_template is not None:
            self.tokenizer.chat_template = model_cfg.chat_template

        # EMA weight averaging is not implemented yet; guard against silent misuse.
        if train_cfg.enable_ema:
            raise NotImplementedError
        else:
            ema_model = None

        # configure optimizer
        policy_optimizer = strategy.create_optimizer(
            policy,
            lr=train_cfg.policy_learning_rate,
            betas=train_cfg.adam_betas,
            weight_decay=train_cfg.l2,
        )

        policy_scheduler = get_scheduler(
            name=train_cfg.policy_lr_scheduler,
            optimizer=policy_optimizer,
            num_warmup_steps=int(
                train_cfg.policy_lr_warmup_steps
            ),  # we dont use ratio math.ceil(max_steps * train_cfg.policy_lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": train_cfg.policy_min_learning_rate},
        )

        if train_cfg.gradient_checkpointing:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": train_cfg.gradient_checkpointing_use_reentrant
                }
            )

        # prepare models/optimizers ... on deepspeed
        self.policy, self.policy_optimizer, self.policy_scheduler = strategy.prepare(
            (policy, policy_optimizer, policy_scheduler),
            is_rlhf=True,
        )

        if train_cfg.enable_ema:
            raise NotImplementedError
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # log, checkpoint, and save model path
        self.save_hf_ckpt = log_cfg.save_hf_ckpt
        self.disable_ds_ckpt = log_cfg.disable_ds_ckpt
        self.ckpt_path = Path(log_cfg.ckpt_path)
        self.save_path = Path(log_cfg.save_path)  # hf model save path
        self.max_ckpt_num = log_cfg.max_ckpt_num
        self.max_ckpt_mem = log_cfg.max_ckpt_mem

        # load policy checkpoint
        self.checkpoint_states = {}
        self.policy_ckpt_path = policy_ckpt_path = self.ckpt_path / "_policy"
        if log_cfg.load_checkpoint and policy_ckpt_path.exists(policy_ckpt_path):
            strategy.print(f"Loading the policy checkpoint from {policy_ckpt_path}")
            _, states = strategy.load_ckpt(self.policy.model, policy_ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states[
                "data_loader_state_dict"
            ]

        # initial offload
        if train_cfg.deepspeed_enable_sleep:
            offload_deepspeed_states(self.policy.model)

        # policy trainer
        self.trainer = PolicyTrainer(
            strategy=strategy,
            policy=self.policy,
            tokenizer=self.tokenizer,
            vllm_engines=self.vllm_engines,
            optimizer=self.policy_optimizer,
            scheduler=self.policy_scheduler,
            train_cfg=self.train_cfg,
            ema_model=self.ema_model,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_masks: Optional[Union[int, list[int]]] = None,
        attention_masks: Optional[torch.Tensor] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()

        # temporarily switch to eval mode for the log-prob forward pass
        self.policy.eval()
        with torch.no_grad():
            action_log_probs = self.policy(
                sequences.to(device),
                action_masks.to(device),
                attention_masks.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        self.policy.train()  # reset model state

        return action_log_probs.to("cpu")

    def append(self, experience: Experience):
        """Add a rollout experience to the trainer's replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def get_replay_buffer(self):
        """Return all experiences currently stored in the replay buffer."""
        return self.trainer.replay_buffer.items

    def fit(self, kl_ctl: float = 0):
        """Train policy model with the replay buffer."""
        torch.cuda.empty_cache()
        self.policy.train()

        status = self.trainer.fit(kl_ctl=kl_ctl)

        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def broadcast_to_vllm(self):
        """Sync the current policy weights to the vLLM rollout engines."""
        self.trainer._broadcast_to_vllm()

    def get_checkpoint_states(self):
        # return states for broadcasting to vllm
        return self.checkpoint_states

    def reload_states(self):
        """Reload DeepSpeed states (e.g. optimizer/parameters) back onto the GPU."""
        reload_deepspeed_states(self.policy.model)

    def offload_states(self):
        """Offload DeepSpeed states off the GPU to free memory."""
        offload_deepspeed_states(self.policy.model)

    def save_model(self):
        # save model checkpoint after fitting on only rank 0
        self.strategy.save_model(
            self.ema_model if self.train_cfg.enable_ema else self.policy,
            self.tokenizer,
            self.save_path,
        )

    def save_checkpoint(self, tag: str, client_states: Dict):
        """Save a DeepSpeed training checkpoint (and optionally an HF-format copy) under the given tag."""
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                model=self.policy.model,
                save_dir=self.policy_ckpt_path,
                tag=tag,
                max_num=self.max_ckpt_num,
                max_mem=self.max_ckpt_mem,
                client_state=client_states,
            )
        if self.save_hf_ckpt:
            save_path = self.policy_ckpt_path / f"hf_{tag}"
            self.strategy.save_model(
                self.ema_model if self.train_cfg.enable_ema else self.policy,
                self.tokenizer,
                save_path,
            )

        # sync
        torch_dist_barrier_and_cuda_sync()
