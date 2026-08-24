"""Policy-model DeepSpeed actor: runs the minibatch policy-gradient updates.

Consumes reward-scored :class:`~src.data.Experience` batches from the replay
buffer and applies the clipped PPO/GRPO/DAPO policy loss (:class:`PolicyLoss`)
plus optional KL regularization, then broadcasts refreshed weights to the vLLM
rollout engines. All optimization is masked to response tokens, so RECAP's
counter-aligned prefix never receives gradients.
"""

import logging
import os
import socket
from abc import ABC
from typing import Dict, List, Optional

import ray
import torch
import torch.distributed
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ray.utils import get_physical_gpu_id
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import (
    init_process_group,
    torch_dist_barrier_and_cuda_sync,
)
from ray.actor import ActorHandle
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import deepspeed
from src.data import Experience, NaiveReplayBuffer

from .loss import PolicyLoss

logger = logging.getLogger(__name__)


class PolicyTrainer(ABC):
    def __init__(
        self,
        strategy: DeepspeedStrategy,
        policy: Actor,
        tokenizer: AutoTokenizer,
        vllm_engines: List[ActorHandle],
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_cfg: DictConfig,
        ema_model: Optional[Actor] = None,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vllm_engines = vllm_engines

        # moving average
        self.ema_model = ema_model
        self.ema_beta = train_cfg.ema_beta  # moving average

        # hyperparams from cfg
        self.train_cfg = train_cfg
        self.max_epochs = train_cfg.max_epochs
        self.dataloader_pin_memory = train_cfg.dataloader_pin_memory
        self.micro_train_batch_size = (
            train_cfg.micro_train_batch_size  # policy training batch size per gpu
        )
        self.aux_loss = train_cfg.aux_loss_coef > 1e-8  # Mixtral 8x7b
        self.entropy_loss_coef = train_cfg.entropy_loss_coef

        # rlhf loss, clipped advantage, for grpo, the kl is not computed here
        self.policy_loss_fn = PolicyLoss(
            clip_eps_low=train_cfg.ratio_clip_eps[0],
            clip_eps_high=train_cfg.ratio_clip_eps[1],
        )

        # creating the policy training dataset
        self.replay_buffer = NaiveReplayBuffer(
            limit=train_cfg.replay_buffer_limit,
            packing_samples=train_cfg.packing_samples,
        )

        # Initialize torch group for weights sync
        # To broadcast model weights from DeepSpeed to vLLM, OpenRLHF creates a single distributed process group that includes:
        #   - DeepSpeed rank 0 (the source of trained weights)
        #   - All vLLM engine workers (the destinations for weight updates)
        backend = train_cfg.vllm_sync_backend
        self.use_cuda_ipc = False
        if (
            backend == "nccl"
            and train_cfg.colocate_all_models
            and not train_cfg.async_train
        ):
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines layer by layer
        # potential issue on master addr difference in torch: https://github.com/OpenRLHF/OpenRLHF/issues/492
        if (
            self.vllm_engines is not None
            and not self.use_cuda_ipc
            and torch.distributed.get_rank() == 0
        ):
            master_address = os.environ.get(
                "MASTER_ADDR", ray._private.services.get_node_ip_address()
            )
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = int(os.environ.get("MASTER_PORT", sock.getsockname()[1]))

            vllm_num_engines = train_cfg.vllm_num_engines
            vllm_tensor_parallel_size = train_cfg.vllm_tensor_parallel_size
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = train_cfg.vllm_sync_with_ray
            group_name = "rlhf_anthony"
            refs = [
                engine.init_process_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=i * vllm_tensor_parallel_size + 1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(
                    world_size=world_size,
                    rank=0,
                    backend=backend,
                    group_name=group_name,
                )
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            ray.get(refs)
        torch_dist_barrier_and_cuda_sync()

    def fit(self, kl_ctl: float):
        should_shuffle = (
            self.strategy.ring_attn_group is None
            and self.strategy.ds_tensor_parallel_size <= 1
        )

        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.micro_train_batch_size,
            shuffle=should_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )

        status_list = []
        status_mean = {}
        device = torch.cuda.current_device()

        # Epoch (inner): Train models multiple times on collected experiences in naive replay buffer
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Policy Training Epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, kl_ctl)

                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]
                status_list.append(status)

                # create shorter status for pbar display
                short_status = {
                    "loss": status["policy_loss"],
                    "kl": status["kl"],
                    "reward": status["reward"],
                    "return": status["return"],
                    "gen_len": status["response_length"],
                    "tot_len": status["total_length"],
                    "lr": status["policy_lr"],
                }

                if "entropy_loss" in status:
                    short_status["ent_loss"] = status["entropy_loss"]

                pbar.set_postfix(short_status)

        if status_list:
            keys = status_list[0].keys()
            status_mean = {
                k: sum(d[k] for d in status_list) / len(status_list) for k in keys
            }

        return status_mean

    def training_step(self, experience: Experience, kl_ctl: float) -> Dict[str, float]:
        """One minibatch policy update.

        Recomputes the current policy's action log-probs, forms the clipped
        surrogate loss against the rollout (old) log-probs and advantages,
        optionally adds a KL-to-reference penalty, and backprops. ``action_masks``
        keeps every term restricted to policy-generated (post-prefill) tokens.
        """
        self.policy.train()

        sequences = experience.sequences
        action_masks = experience.action_masks
        attention_masks = experience.attention_masks
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        # compute current policy logprob
        # in the first step, this is exactly the old_action_logprobs, but in the following steps, they should vary
        action_log_probs, output = self.policy(
            sequences=sequences,
            action_mask=action_masks,
            attention_mask=attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=None,
            return_entropy=self.entropy_loss_coef is not None,
        )

        # loss function, clip_ratio is the percentage of the updates being clipped in ppo and grpo
        policy_loss, clip_ratio, importance_sampling_ratio = self.policy_loss_fn(
            log_probs=action_log_probs,
            old_log_probs=old_action_log_probs,
            advantages=advantages,
            action_mask=action_masks,
        )
        experience.info["policy_clip_ratio"] = clip_ratio
        experience.info["importance_sampling_ratio"] = importance_sampling_ratio

        if self.train_cfg.use_kl_loss:
            if self.train_cfg.init_kl_coef > 0:
                kl = compute_approx_kl(
                    log_probs=action_log_probs,
                    log_probs_base=base_action_log_probs,
                    kl_estimator=self.train_cfg.kl_estimator,
                )
            else:
                kl = torch.zeros_like(
                    action_log_probs,
                    dtype=action_log_probs.dtype,
                    device=action_log_probs.device,
                )
            kl_loss = masked_mean(kl, action_masks)
            experience.info["kl"] = kl_loss.detach()
        else:
            kl_loss = 0

        loss = policy_loss + kl_ctl * kl_loss

        # mixtral
        if self.aux_loss:
            loss += output.aux_loss * self.train_cfg.aux_loss_coef

        # entropy loss
        if self.train_cfg.entropy_loss_coef is not None:
            entropy_loss = masked_mean(
                output.entropy[:, -action_masks.shape[1] :], action_masks
            )
            if self.train_cfg.entropy_loss_coef != 0:
                loss -= self.train_cfg.entropy_loss_coef * entropy_loss

        self.strategy.backward(loss, self.policy, self.optimizer)
        self.strategy.optimizer_step(
            self.optimizer, self.policy, self.scheduler, name="actor"
        )
        if self.ema_model:
            self.strategy.moving_average(
                self.policy, self.ema_model, self.ema_beta, "cuda"
            )

        # status
        status = {
            "policy_loss": policy_loss.detach().item(),
            "policy_lr": self.scheduler.get_last_lr()[0],
        }
        if self.train_cfg.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()

        # merge logs from info field
        for k, v in experience.info.items():
            if isinstance(v, list):
                status[k] = torch.tensor(v, dtype=torch.float).mean().item()
            elif isinstance(v, torch.Tensor):
                status[k] = v.float().mean().item()
        return status

    def _broadcast_to_vllm(self):
        use_prefix_cache = self.train_cfg.enable_prefix_caching
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.policy.model.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(name: str, param: Parameter, count: int, num_params: int):
            use_ray = self.train_cfg.vllm_sync_with_ray  # use ray to broadcast weights
            # Fire all vllm engines for broadcast, calling the method defined in openrlhf.trainer.ray.vllm_worker_wrap.py
            if torch.distributed.get_rank() == 0:
                # ds_shape is DeepSpeed's unsharded shape (for ZeRO-3)
                shape = param.shape if self.strategy.stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name=name,
                        dtype=param.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(
                        param.data, 0, group_name=self._model_update_group
                    )
                else:
                    torch.distributed.broadcast(
                        param.data, 0, group=self._model_update_group
                    )
                ray.get(refs)

        def _handle_cuda_ipc(name: str, param: Parameter, count: int, num_params: int):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name=name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams(
                        [param], model, enabled=True
                    ):
                        _broadcast_param(name, param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters(
                        [param], enabled=self.strategy.stage == 3
                    ):
                        _broadcast_param(name, param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams(
                        [param], model, enabled=True
                    ):
                        _handle_cuda_ipc(name, param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters(
                        [param], enabled=self.strategy.stage == 3
                    ):
                        _handle_cuda_ipc(name, param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()
