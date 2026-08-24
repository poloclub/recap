"""Supervised fine-tuning (SFT) entry point.

Loads a pretrained causal LM via OpenRLHF's ``Actor`` wrapper, blends the
datasets configured under ``cfg.data``, and fine-tunes it with DeepSpeed
using :class:`src.trainer.SFTTrainer`. Optionally computes a KL penalty
against a frozen reference model during training.

Config: Hydra config ``config/sft.yaml`` (``config_name="sft"``).
Invocation: ``deepspeed --module src.sft ++name=<run_name> ...`` (see the
``experiment/sft/%/.done_sft`` target in ``makefiles/sft.mk``).
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import hydra
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from transformers import get_scheduler

from src.data import HydraConfigDataset, blend_datasets
from src.model import get_tokenizer
from src.trainer import SFTTrainer

logger = logging.getLogger(__name__)

# By default, Hydra intercepts and manages CLI arguments, which prevents DeepSpeed's --local_rank from being passed correctly.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args, unknown = parser.parse_known_args()  # Ignore unknown arguments
# Remove `--local_rank` from sys.argv before Hydra processes it
sys.argv = [x for x in sys.argv if "--local_rank" not in x]


@hydra.main(config_path="../config", config_name="sft", version_base="1.3")
def main(cfg: DictConfig):
    """Run supervised fine-tuning using the resolved Hydra config ``cfg``.

    Initializes DeepSpeed distributed training, loads the policy
    model/tokenizer (and a frozen reference model if ``cfg.train.use_kl``),
    builds the train/eval dataloaders, and runs ``SFTTrainer.fit``, saving
    the final model in HF format to ``cfg.log.save_path``.
    """
    output_dir = Path(os.getcwd())  # output to experiment/${name}
    cfg.log.ckpt_path = str(output_dir / cfg.log.ckpt_path)
    cfg.log.save_path = str(output_dir / cfg.log.save_path)

    # ==== init distributed ====
    cfg.deepspeed.local_rank = args.local_rank
    strategy = DeepspeedStrategy(
        seed=cfg.seed,
        full_determinism=cfg.deepspeed.full_determinism,
        max_norm=cfg.train.max_norm,
        micro_train_batch_size=cfg.train.micro_train_batch_size,
        train_batch_size=cfg.train.train_batch_size,
        zero_stage=cfg.deepspeed.zero_stage,
        bf16=cfg.model.bf16,
        args=cfg.deepspeed,
    )
    strategy.setup_distributed()
    world_size = dist.get_world_size()
    if strategy.is_rank_0():
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))
        logger.info(f"Distributed training world size={world_size} ...")

    # load HF model and tokenizer
    if strategy.is_rank_0():
        logger.info(f"Loading training model and tokenizer from {cfg.model.name} ...")
    model = Actor(
        pretrain_or_model=cfg.model.name,
        use_flash_attention_2=cfg.model.flash_attn,
        bf16=cfg.model.bf16,
        load_in_4bit=cfg.model.load_in_4bit,
        lora_rank=cfg.model.lora.rank,
        lora_alpha=cfg.model.lora.alpha,
        target_modules=cfg.model.lora.target_modules,
        lora_dropout=cfg.model.lora.dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=cfg.train.packing_samples,
    )
    tokenizer = get_tokenizer(
        pretrain=cfg.model.name,
        model=None,
        padding_side="left",
        use_fast=not cfg.model.disable_fast_tokenizer,
    )
    # overwrite incorrect deepseek model chat template
    if cfg.model.chat_template is not None:
        if strategy.is_rank_0():
            logger.info("Overwritting default chat template ...")
        tokenizer.chat_template = cfg.model.chat_template

    # load reference model weights
    if cfg.train.use_kl:
        if strategy.is_rank_0():
            logger.info(
                f"Loading reference model {cfg.ref_model.name} for dynamic safety training ..."
            )
        ref_model = Actor(
            pretrain_or_model=cfg.ref_model.name,
            use_flash_attention_2=cfg.ref_model.flash_attn,
            bf16=cfg.ref_model.bf16,
            load_in_4bit=cfg.ref_model.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=cfg.train.ref_offload),
            packing_samples=cfg.train.packing_samples,
        )
        if cfg.train.ref_offload:
            ref_model._offload = True

    # gradient checkpointing
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": cfg.train.gradient_checkpointing_use_reentrant
            }
        )

    # optimizer
    optimizer = strategy.create_optimizer(
        model,
        lr=cfg.train.learning_rate,
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.l2,
    )

    # dataset
    assert not cfg.data.use_prefill, "SFT does not support prefilling"
    train_data, eval_data = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)

    max_length = cfg.train.prompt_max_len + cfg.train.generate_max_len
    if strategy.is_rank_0():
        logger.info(
            f"Setting SFT max_length={max_length} (prompt: {cfg.train.prompt_max_len}, generation: {cfg.train.generate_max_len}) ..."
        )

    train_dataset = HydraConfigDataset(
        data=train_data,
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        tokenizer=tokenizer,
        max_length=max_length,
        multiturn=cfg.train.multiturn,
        pretrain_mode=cfg.train.use_pretrain_loss,
        for_guardrail=False,
        use_response=True,
        use_value=cfg.train.use_value,
    )

    train_dataloader = strategy.setup_dataloader(
        replay_buffer=train_dataset,
        batch_size=cfg.train.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.packing_collate_fn
        if cfg.train.packing_samples
        else train_dataset.collate_fn,
    )

    # eval data processing
    if cfg.data.use_eval:
        eval_dataset = HydraConfigDataset(
            data=eval_data,
            data_cfg=cfg.data,
            model_cfg=cfg.model,
            tokenizer=tokenizer,
            max_length=max_length,
            multiturn=cfg.train.multiturn,
            pretrain_mode=cfg.train.use_pretrain_loss,
            for_guardrail=False,
            use_response=True,
            use_value=cfg.train.use_value,
        )

        eval_dataloader = strategy.setup_dataloader(
            replay_buffer=eval_dataset,
            batch_size=cfg.train.micro_train_batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=eval_dataset.packing_collate_fn
            if cfg.train.packing_samples
            else eval_dataset.collate_fn,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // cfg.train.train_batch_size
    max_steps = math.ceil(cfg.train.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        name=cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * cfg.train.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": cfg.train.min_learning_rate},
    )

    # prepare models
    if cfg.train.use_kl:
        (model, optimizer, scheduler), ref_model = strategy.prepare(
            (model, optimizer, scheduler), ref_model
        )
    else:
        model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler))

    # load checkpoint
    consumed_samples = 0
    if cfg.log.load_checkpoint and os.path.exists(cfg.log.ckpt_path):
        _, states = strategy.load_ckpt(model.model, cfg.log.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(
            f"Loaded the checkpoint: {cfg.log.ckpt_path}, consumed_samples: {consumed_samples}"
        )

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        tokenizer=tokenizer,
        train_cfg=cfg.train,
        log_cfg=cfg.log,
        ref_model=ref_model if cfg.train.use_kl else None,
    )

    if strategy.is_rank_0():
        logger.info("Training starts ...")
    trainer.fit(consumed_samples, num_update_steps_per_epoch)

    if strategy.is_rank_0():
        logger.info(f"Saving final model in HF format to {cfg.log.save_path} ...")
    strategy.save_model(model, tokenizer, cfg.log.save_path)

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
