"""
This file wraps all necessary models in rlhf with ray actor
1. openrlhf ray launcher.py
"""

# Ray actor for the frozen reference (SFT/initial) policy: it only produces
# log-probs, which are used to compute the KL penalty against the trainable
# policy during RLHF.

import logging
from typing import Optional

import ray
import torch
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy

from .ray_actor_group import BaseModelActor

logger = logging.getLogger(__name__)


__all__ = ["ReferenceModelActor"]


@ray.remote(num_gpus=1)
class ReferenceModelActor(BaseModelActor):
    """we only move tensors to gpu inside forward, so calling empty_cache after calling forward will release the gpu memory
    but deepspeed initialization still takes gpu memory to store the model weights
    """

    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, model_cfg: DictConfig, train_cfg: DictConfig
    ):
        """Load the frozen reference model under DeepSpeed and set it to eval mode."""
        self._setup_distributed(strategy)
        model = Actor(
            pretrain_or_model=model_cfg.name,
            use_flash_attention_2=model_cfg.flash_attn,
            bf16=model_cfg.bf16,
            load_in_4bit=model_cfg.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=train_cfg.offload_ref),
            packing_samples=train_cfg.packing_samples,
            temperature=model_cfg.generation_args.temperature,
            use_liger_kernel=train_cfg.use_liger_kernel,
        )

        if strategy.is_rank_0():
            logger.info(f"Reference model:\n{model}")

        if train_cfg.offload_ref:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_masks: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Compute reference log-probs for the given sequences (used for the RLHF KL penalty)."""
        device = torch.cuda.current_device()

        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_masks.to(device),
                attention_masks.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")
