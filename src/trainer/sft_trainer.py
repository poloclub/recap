"""Supervised fine-tuning (SFT) training loop.

Trains a causal LM with a next-token loss over completion tokens only (prompt
tokens are masked out), optionally weighted per-token values and/or a KL term
against a frozen reference model.
"""

import logging
import os
from abc import ABC

import torch
from omegaconf import DictConfig
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_sampler import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from src.model.loss import ValueWeightedGPTLMLoss

logger = logging.getLogger(__name__)


class SFTTrainer(ABC):
    """Runs the SFT training/eval loop over a pre-tokenized dataloader,
    handling loss masking, logging, checkpointing, and periodic evaluation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        strategy: DeepspeedStrategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        scheduler: _LRScheduler,
        tokenizer: AutoTokenizer,
        train_cfg: DictConfig,
        log_cfg: DictConfig,
        ref_model: PreTrainedModel = None,
    ) -> None:
        self.strategy = strategy
        self.train_cfg = train_cfg
        self.log_cfg = log_cfg
        self.epochs = train_cfg.max_epochs
        self.max_norm = train_cfg.max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = train_cfg.use_pretrain_loss
        self.model = model
        self.ref_model = ref_model
        if ref_model is not None:
            self.ref_model.eval()
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.save_hf_ckpt = log_cfg.save_hf_ckpt
        self.disable_ds_ckpt = log_cfg.disable_ds_ckpt

        self.loss_fn = ValueWeightedGPTLMLoss(
            use_value=train_cfg.use_value,
            use_kl=train_cfg.use_kl,
            ring_attn_group=self.strategy.ring_attn_group,
            kl_estimator=train_cfg.kl_estimator,
            kl_scale=train_cfg.kl_scale,
        )

        # Mixtral 8*7b
        self.aux_loss = train_cfg.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = train_cfg.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if log_cfg.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=log_cfg.use_wandb)
            wandb.init(
                entity=log_cfg.wandb_org,
                project=log_cfg.wandb_project,
                group=log_cfg.wandb_group,
                name=log_cfg.wandb_run_name,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True
            )
            wandb.define_metric("eval/global_step")
            wandb.define_metric(
                "eval/*", step_metric="eval/global_step", step_sync=True
            )

        # Initialize TensorBoard writer if wandb is not available
        if (
            log_cfg.use_tensorboard
            and self._wandb is None
            and self.strategy.is_rank_0()
        ):
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_cfg.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(log_cfg.use_tensorboard, log_cfg.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, consumed_samples: int = 0, num_update_steps_per_epoch=None):
        """Run the training loop across epochs, resuming from `consumed_samples`
        (e.g. after a restart) and periodically logging/evaluating/checkpointing.
        """
        # get eval and save steps
        if self.train_cfg.eval_steps == -1:
            self.train_cfg.eval_steps = (
                num_update_steps_per_epoch  # Evaluate once per epoch
            )
        if self.log_cfg.save_steps == -1:
            self.log_cfg.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = (
            consumed_samples
            // self.train_cfg.train_batch_size
            * self.strategy.accumulated_gradient
            + 1
        )
        start_epoch = (
            consumed_samples
            // self.train_cfg.train_batch_size
            // num_update_steps_per_epoch
        )
        consumed_samples = consumed_samples % (
            num_update_steps_per_epoch * self.train_cfg.train_batch_size
        )

        loss_sum = 0
        for epoch in tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        ):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch,
                    consumed_samples=0 if epoch > start_epoch else consumed_samples,
                )

            step_bar = tqdm(
                self.train_dataloader,
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for prompt_id_lens, input_tokens, infos in step_bar:
                inputs, attention_masks = (
                    input_tokens["input_ids"],
                    input_tokens["attention_mask"],
                )
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_masks = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_masks = attention_masks.to(
                        torch.cuda.current_device()
                    ).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(
                        inputs, attention_mask=attention_masks, return_output=True
                    )
                    if self.train_cfg.use_kl:
                        with torch.no_grad():
                            ref_output = self.ref_model(
                                inputs,
                                attention_mask=attention_masks,
                                return_output=True,
                                ring_attn_group=self.strategy.ring_attn_group,
                            )
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_masks,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                    if self.train_cfg.use_kl:
                        with torch.no_grad():
                            ref_output = self.ref_model(
                                inputs,
                                attention_mask=attention_masks,
                                return_output=True,
                                ring_attn_group=self.strategy.ring_attn_group,
                                packed_seq_lens=infos["input_length"],
                            )

                # loss function
                # labels default to the input ids (next-token prediction), with padded
                # positions ignored via IGNORE_INDEX
                labels = torch.where(
                    attention_masks.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                # unless doing plain pretrain-style LM loss, mask out prompt tokens so the
                # loss is only computed on the completion (response) tokens
                if not self.pretrain_mode:
                    if self.packing_samples:
                        # As response_ranges need to constrain the dataset organization strictly, we handle multiturn feature separately.
                        if infos["response_ranges"]:
                            # multiturn: only the explicit assistant-turn spans count as labels
                            dump_labels = torch.full(
                                labels.size(), self.loss_fn.IGNORE_INDEX
                            ).to(labels.device)
                            for response_ranges in infos["response_ranges"]:
                                for response_range in response_ranges:
                                    dump_labels[0][
                                        response_range[0] : response_range[1] + 1
                                    ] = labels[0][
                                        response_range[0] : response_range[1] + 1
                                    ]
                            labels = dump_labels
                        else:
                            # single-turn packed samples: mask the prompt prefix of each
                            # packed segment using its recorded length
                            index = 0
                            for input_length, source_len in zip(
                                infos["input_length"], prompt_id_lens
                            ):
                                labels[0][index : index + source_len + 1] = (
                                    self.loss_fn.IGNORE_INDEX
                                )
                                index += input_length
                    else:
                        # unpacked samples: mask the prompt prefix of each sample in the batch
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                gpt_loss = self.loss_fn(
                    logits=output.logits,
                    labels=labels,
                    values=infos["value"] if self.train_cfg.use_value else None,
                    positions=infos["position"] if self.train_cfg.use_value else None,
                    ref_logits=ref_output.logits if self.train_cfg.use_kl else None,
                )

                loss = gpt_loss[0] + aux_loss * self.train_cfg.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += gpt_loss[0].item()
                logs_dict = {
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.train_cfg.use_value:
                    logs_dict["total_loss"] = gpt_loss[0].item()
                    logs_dict["ce_loss"] = gpt_loss[1].item()
                    if self.train_cfg.use_kl:
                        logs_dict["kl_loss"] = gpt_loss[2].item()
                else:
                    if self.train_cfg.use_kl:
                        logs_dict["kl_loss"] = gpt_loss[0].item()
                    else:
                        logs_dict["ce_loss"] = gpt_loss[0].item()

                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = (
                        loss_sum / self.strategy.accumulated_gradient
                    )
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {
                        "consumed_samples": global_step
                        * self.train_cfg.train_batch_size
                    }
                    self.save_logs_and_checkpoints(
                        global_step, logs_dict, client_states
                    )

                step += 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, global_step, logs_dict={}, client_states={}):
        """At the configured intervals: log metrics (wandb/tensorboard), run
        evaluation, and save a checkpoint (deepspeed and/or HF format)."""
        if global_step % self.log_cfg.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {**logs_dict, "global_step": global_step}.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            self.train_cfg.eval_steps > 0
            and global_step % self.train_cfg.eval_steps == 0
        ):
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                if self.strategy.is_rank_0():
                    logger.info("==== Start evaluation ====")
                self.evaluate(global_step)
            else:
                if self.strategy.is_rank_0():
                    logger.info("==== Skip evaluation due to insufficient examples ====")

        # save ckpt
        if global_step % self.log_cfg.save_steps == 0:
            if self.strategy.is_rank_0():
                logger.info(f"Saving checkpoints to {self.log_cfg.ckpt_path} ...")
            tag = f"global_step_{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model,
                    self.log_cfg.ckpt_path,
                    tag,
                    self.log_cfg.max_ckpt_num,
                    self.log_cfg.max_ckpt_mem,
                    client_states,
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(self.log_cfg.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, steps: int = 0):
        """Run a no-grad pass over the eval dataloader and log the mean loss."""
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                self.eval_dataloader,
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompt_id_lens, input_tokens, infos in step_bar:
                inputs, attention_masks = (
                    input_tokens["input_ids"],
                    input_tokens["attention_mask"],
                )
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_masks = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_masks = attention_masks.to(
                        torch.cuda.current_device()
                    ).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(
                        inputs, attention_mask=attention_masks, return_output=True
                    )
                    if self.train_cfg.use_kl:
                        ref_output = self.ref_model(
                            inputs,
                            attention_mask=attention_masks,
                            return_output=True,
                            ring_attn_group=self.strategy.ring_attn_group,
                        )
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_masks,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                    if self.train_cfg.use_kl:
                        ref_output = self.ref_model(
                            inputs,
                            attention_mask=attention_masks,
                            return_output=True,
                            ring_attn_group=self.strategy.ring_attn_group,
                            packed_seq_lens=infos["input_length"],
                        )

                # loss function
                labels = torch.where(
                    attention_masks.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        if infos["response_ranges"]:
                            dump_labels = torch.full(
                                labels.size(), self.loss_fn.IGNORE_INDEX
                            ).to(labels.device)
                            for response_ranges in infos["response_ranges"]:
                                for response_range in response_ranges:
                                    dump_labels[0][
                                        response_range[0] : response_range[1]
                                    ] = labels[0][response_range[0] : response_range[1]]
                            labels = dump_labels
                        else:
                            index = 0
                            for input_length, source_len in zip(
                                infos["input_length"], prompt_id_lens
                            ):
                                labels[0][index : index + source_len] = (
                                    self.loss_fn.IGNORE_INDEX
                                )
                                index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(
                    logits=output.logits,
                    labels=labels,
                    values=infos["value"] if self.train_cfg.use_value else None,
                    positions=infos["position"] if self.train_cfg.use_value else None,
                    ref_logits=ref_output.logits if self.train_cfg.use_kl else None,
                )[0]

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {
                        "eval/%s" % k: v
                        for k, v in {**logs, "global_step": steps}.items()
                    }
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
