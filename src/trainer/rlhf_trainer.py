"""Online RLHF trainer: the central controller for the GRPO/DAPO training loop.

:class:`RLHFTrainer` owns the per-step cycle used for RECAP training:

    rollout (vLLM) -> make experiences + score rewards -> optional DAPO dynamic
    sampling -> push to replay buffer -> update policy (and critic) -> log/ckpt

Counter-aligned prefilling needs no special handling here: the prefix is baked
into the tokenized prompt by :mod:`src.data.dataset`, and ``prompt_ids_len``
marks where the policy's (optimized) tokens begin, so the standard loop trains
prefilled and standard prompts identically.
"""

import logging
import math
import os
import time
from abc import ABC
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import ray
import torch
from omegaconf import DictConfig
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.utils.deepspeed import DeepspeedStrategy
from ray.actor import ActorHandle
from tqdm import tqdm

from src.data import HydraConfigDataset, blend_datasets
from src.model import RayActorGroup, batch_vllm_engine_call, get_tokenizer

from .rollout import ExperienceMaker
from .util import get_dynamic_sampling_experiences

logger = logging.getLogger(__name__)

__all__ = ["RLHFTrainer"]


class BaseRLHFTrainer(ABC):
    """Shared setup for RLHF trainers.

    Holds the policy/critic/reward/reference Ray actor groups and the vLLM
    engines, reads all training hyperparameters off the Hydra config, builds the
    train/eval dataloaders, and provides the reusable pieces of a step:
    critic+policy update, weight broadcast to vLLM, and logging/checkpointing.
    Concrete loop logic lives in :class:`RLHFTrainer.fit`.
    """

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        policy_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        multi_reward_model_group: Union[RayActorGroup, List[RayActorGroup]],
        ref_model_group: RayActorGroup,
        vllm_engines: List[ActorHandle],
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.cfg = cfg
        self.policy_cfg = policy_cfg = cfg.model.policy_model
        self.train_cfg = train_cfg = cfg.train
        self.log_cfg = log_cfg = cfg.log
        self.data_cfg = cfg.data

        self.policy_tokenizer = get_tokenizer(
            pretrain=policy_cfg.name,
            model=None,
            padding_side="left",
            use_fast=not policy_cfg.disable_fast_tokenizer,
        )
        # overwrite incorrect deepseek model chat template
        if policy_cfg.chat_template is not None:
            self.policy_tokenizer.chat_template = policy_cfg.chat_template

        self.policy_model_group = policy_model_group
        self.critic_model_group = critic_model_group
        self.multi_reward_model_group = multi_reward_model_group
        self.ref_model_group = ref_model_group
        self.vllm_engines = vllm_engines
        self.remote_reward_model = None

        # configs
        self.max_epochs = train_cfg.max_epochs
        self.remote_rm_url = train_cfg.remote_rm_url
        self.init_kl_coef = train_cfg.init_kl_coef
        self.kl_target = train_cfg.kl_target
        self.kl_horizon = train_cfg.kl_horizon
        self.freezing_policy_steps = train_cfg.freezing_policy_steps
        self.num_episodes = train_cfg.num_episodes
        self.dynamic_sampling = train_cfg.dynamic_sampling
        self.micro_rollout_batch_size = train_cfg.micro_rollout_batch_size
        self.vllm_generate_batch_size = train_cfg.vllm_generate_batch_size
        self.train_batch_size = train_cfg.train_batch_size
        self.n_samples_per_prompt = train_cfg.n_samples_per_prompt
        self.vllm_enable_sleep = train_cfg.vllm_enable_sleep
        self.max_training_steps = (
            float("inf")
            if train_cfg.max_training_steps <= 0
            else train_cfg.max_training_steps
        )

        self.eval_temperature = train_cfg.eval_temperature
        self.eval_n_samples_per_prompt = train_cfg.eval_n_samples_per_prompt
        self.eval_steps = (
            float("inf")
            if train_cfg.eval_steps == -1
            else train_cfg.eval_steps  # do not evaluate if set to -1
        )
        if log_cfg.save_steps == -1:
            self.save_steps = float("inf")  # disables checkpointing
        else:
            self.save_steps = log_cfg.save_steps
        self.next_save_step = self.save_steps

        if train_cfg.agent_func_path is not None:
            raise NotImplementedError
        else:
            from .rollout import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None

        if self.log_cfg.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.log_cfg.use_wandb)
            wandb.init(
                entity=self.log_cfg.wandb_org,
                project=self.log_cfg.wandb_project,
                group=self.log_cfg.wandb_group,
                name=self.log_cfg.wandb_run_name,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True
            )
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            self.generated_samples_table = wandb.Table(
                columns=["global_step", "text", "reward"]
            )

        # Initialize TensorBoard writer if wandb is not available
        if (
            self.log_cfg.use_tensorboard
            and self._wandb is None
            and self.strategy.is_rank_0()
        ):
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.log_cfg.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.log_cfg.use_tensorboard, self.log_cfg.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def prepare_datasets(self):
        train_data, eval_data = blend_datasets(
            datasets_cfg=self.data_cfg, seed=self.cfg.seed
        )
        train_dataset = HydraConfigDataset(
            data=train_data,
            data_cfg=self.data_cfg,
            model_cfg=self.policy_cfg,
            tokenizer=self.policy_tokenizer,
            max_length=self.train_cfg.prompt_max_len,  # because we only tokenize prompt, use to truncate long prompt
            for_guardrail=False,
            use_response=False,
            use_value=False,
        )
        self.train_dataset = train_dataset

        # we need to send the tokenized prompt to the sample generator, and drop padding in collate func
        self.train_dataloader = self.strategy.setup_dataloader(
            replay_buffer=train_dataset,
            batch_size=self.vllm_generate_batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=train_dataset.packing_collate_fn
            if self.train_cfg.packing_samples
            else partial(train_dataset.collate_fn, padding=False),
        )

        if self.data_cfg.use_eval:
            eval_dataset = HydraConfigDataset(
                data=eval_data,
                data_cfg=self.data_cfg,
                model_cfg=self.policy_cfg,
                tokenizer=self.policy_tokenizer,
                max_length=self.train_cfg.prompt_max_len,
                for_guardrail=False,
                use_response=False,
                use_value=False,
            )
            self.eval_dataset = eval_dataset

            self.eval_dataloader = self.strategy.setup_dataloader(
                replay_buffer=eval_dataset,
                batch_size=math.ceil(
                    self.vllm_generate_batch_size
                    * len(eval_dataset)
                    / len(train_dataset)
                ),
                pin_memory=True,
                shuffle=False,
                collate_fn=eval_dataset.packing_collate_fn
                if self.train_cfg.packing_samples
                else partial(eval_dataset.collate_fn, padding=False),
            )
        else:
            self.eval_dataloader = None

        # total training steps of the policy model
        dataset_max_steps = (
            len(train_dataset)
            * self.n_samples_per_prompt
            // self.train_batch_size
            * self.num_episodes
            * self.max_epochs
        )

        self.max_steps = int(min(self.max_training_steps, dataset_max_steps))
        logger.info(f"Setting max training steps as {self.max_steps} ...")

    def get_max_steps(self):
        return self.max_steps

    def train_policy_and_value(self, global_step: int) -> Dict:
        status = {}
        start_time = time.time()
        deepspeed_enable_sleep = self.train_cfg.deepspeed_enable_sleep
        colocate_all_models = self.train_cfg.colocate_all_models

        # === train critic model
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if deepspeed_enable_sleep:
                ray.get(
                    self.critic_model_group.async_run_method(
                        method_name="reload_states"
                    )
                )

            critic_status_ref = self.critic_model_group.async_run_method(
                method_name="fit"
            )

            if colocate_all_models or deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if deepspeed_enable_sleep:
                ray.get(
                    self.critic_model_group.async_run_method(
                        method_name="offload_states"
                    )
                )

        # === train policy model
        if global_step > self.freezing_policy_steps:
            if deepspeed_enable_sleep:
                self.policy_model_group.async_run_method(method_name="reload_states")

            policy_status_ref = self.policy_model_group.async_run_method(
                method_name="fit", kl_ctl=self.kl_ctl.value
            )
            status.update(
                ray.get(policy_status_ref)[0]
            )  # status are averaged over all ray actors

            if deepspeed_enable_sleep:
                self.policy_model_group.async_run_method(method_name="offload_states")

            # broadcast weights from deepspeed to vllm engines
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()

        # === wait for training the critic model
        if (
            self.critic_model_group
            and not colocate_all_models
            and not deepspeed_enable_sleep
        ):
            status.update(ray.get(critic_status_ref)[0])

        duration = time.time() - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"==== Policy training completed in {time_str} (HH:MM:SS) ====")

        return status

    def _broadcast_to_vllm(self):
        # broadcast deepspeed trained model weights to vllm, runtime vllm weights update: https://github.com/vllm-project/vllm/issues/5723
        if self.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(
            self.policy_model_group.async_run_method(method_name="broadcast_to_vllm")
        )

        if self.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(
        self, global_step: int, logs_dict: Dict, client_states: Dict
    ):
        if global_step % self.log_cfg.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                # Add generated samples to wandb using Table
                table_log = None
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns,
                        data=self.generated_samples_table.data,
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    table_log = {"train/generated_samples": new_table}
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if table_log:
                    logs.update(table_log)
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text(
                            "train/generated_samples", formatted_text, global_step
                        )
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # save checkpoints
        if global_step >= self.next_save_step:
            tag = f"global_step_{global_step}"
            ref = self.policy_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(
                    self.critic_model_group.async_run_method(
                        method_name="save_checkpoint", tag=tag
                    )
                )
            ray.get(ref)
            self.next_save_step += self.save_steps

    @torch.no_grad()
    def evaluate(self, global_step: int):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        # First collect all prompts and labels
        all_prompts = []
        all_labels = []
        prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

        pass


class RLHFTrainer(BaseRLHFTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        policy_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        multi_reward_model_group: Union[RayActorGroup, List[RayActorGroup]],
        ref_model_group: RayActorGroup,
        vllm_engines: List[ActorHandle],
        cfg: DictConfig,
    ) -> None:
        super().__init__(
            strategy,
            policy_model_group,
            critic_model_group,
            multi_reward_model_group,
            ref_model_group,
            vllm_engines,
            cfg,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(
                self.init_kl_coef, self.kl_target, self.kl_horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.remote_rm_url and not self.remote_rm_url[0] == "agent":
            raise NotImplementedError
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(
                self.args, self.remote_rm_url
            )

        self.samples_generator = self.generator_cls(
            vllm_engines=self.vllm_engines,
            tokenizer=self.policy_tokenizer,
            generation_cfg=self.policy_cfg.generation_args,
            train_cfg=self.train_cfg,
            apply_chat_template=self.data_cfg.apply_chat_template,
            remote_reward_model=self.remote_reward_model,
        )

        self.experience_maker = ExperienceMaker(
            policy_model_group=self.policy_model_group,
            critic_model_group=self.critic_model_group,
            multi_reward_model_group=self.multi_reward_model_group,
            ref_model_group=self.ref_model_group,
            policy_tokenizer=self.policy_tokenizer,
            kl_controller=self.kl_ctl,
            train_cfg=self.train_cfg,
            ring_attn_size=self.policy_cfg.ring_attn_size,
            ds_tensor_parallel_size=cfg.deepspeed.ds_tensor_parallel_size,
            remote_reward_model=self.remote_reward_model,
        )

        self._init_wandb()
        self.prepare_datasets()

    def fit(self):
        """Run the online RL loop until max steps / episodes are reached.

        Per iteration: (1) roll out completions from vLLM for a batch of prompts
        (prompts may carry a counter-aligned prefill), (2) build reward-scored
        experiences, (2.1) optionally apply DAPO dynamic sampling to drop prompt
        groups whose rollouts are all-good or all-bad, (3) append to the replay
        buffer, (4) update the policy (and critic), (5) log and checkpoint.
        """
        # broadcast init checkpoint to vllm
        ckpt_path = Path(self.log_cfg.ckpt_path) / "_policy"
        if self.log_cfg.load_checkpoint and ckpt_path.exists(ckpt_path):
            ckpt_states = ray.get(
                self.policy_model_group.async_run_method(
                    method_name="get_checkpoint_states"
                )
            )[0]
            logger.info(f"Checkpoint States: {ckpt_states}")
            self._broadcast_to_vllm()
        else:
            ckpt_states = {
                "global_step": 0,
                "episode": 0,
                "data_loader_state_dict": {},
            }

        # restore step and start training
        global_step = ckpt_states["global_step"]
        start_episode = ckpt_states["episode"]
        data_loader_state_dict = ckpt_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.train_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(start_episode, self.num_episodes):
            if global_step >= self.max_steps:
                logger.info(
                    f"Current global step: {global_step} hit max training steps: {self.max_steps}"
                )
                break

            pbar = tqdm(
                self.train_dataloader,
                desc=f"Episode [{episode + 1}/{self.num_episodes}]",
                disable=False,
            )

            filtered_samples = []
            number_of_samples = 0
            for prompt_ids_lens, input_tokens, info in pbar:
                # step 1: rollout
                rollout_samples = self.samples_generator.generate_samples(
                    all_tokenized_prompts=input_tokens, info=info
                )

                # step 2: format into List[Experience]
                experiences = self.experience_maker.make_experience_batch(
                    rollout_samples
                )

                # step 2.1: dynamic sampling from dapo (optional)
                pass_rate = None
                if self.dynamic_sampling:
                    number_of_samples += len(experiences)
                    min_reward, max_reward = (
                        self.train_cfg.dynamic_sampling_reward_range
                    )
                    kept_samples = get_dynamic_sampling_experiences(
                        experiences=experiences,
                        n_samples_per_prompt=self.n_samples_per_prompt,
                        min_reward=min_reward,
                        max_reward=max_reward,
                    )
                    if kept_samples:
                        filtered_samples.extend(kept_samples)

                    # Continue sampling if filtered samples are insufficient
                    minimum_samples = (
                        self.vllm_generate_batch_size * self.n_samples_per_prompt
                    )
                    if len(filtered_samples) < minimum_samples:
                        logger.info(
                            f"Filtered_samples {len(filtered_samples)} < minimum rollout prompts {minimum_samples}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    experiences = filtered_samples[:minimum_samples]
                    filtered_samples = []
                    number_of_samples = 0

                # step 3: save to replay buffer
                refs = self.policy_model_group.async_run_method_batch(
                    method_name="append", experience=experiences
                )
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(
                            method_name="append", experience=experiences
                        )
                    )
                ray.get(refs)

                # step 4: train policy
                status = self.train_policy_and_value(global_step=global_step)
                if "kl" in status:
                    self.kl_ctl.update(
                        status["kl"],
                        self.micro_rollout_batch_size * self.n_samples_per_prompt,
                    )

                # Add generated samples to status dictionary
                if self.dynamic_sampling:
                    status["dynamic_sampling_pass_rate"] = pass_rate
                logger.info(f"✨ ==== Global step {global_step}: {status} ====")

                # step 5: checkpoint & evaluation
                # save one prompt + model generation + reward
                wandb_row_text = "PROMPT: {}\n\nRESPONSE: {}".format(
                    experiences[0].conversations[0]["content"],
                    experiences[0].generation_texts,
                )
                if experiences[0].conversations[-1]["role"] == "assistant":
                    wandb_row_text += "\n\nLABEL: {}".format(
                        experiences[0].conversations[-1]["content"]
                    )
                status["generated_samples"] = [
                    wandb_row_text,
                    experiences[0].info["reward"],
                ]
                client_states = {
                    "global_step": global_step,
                    "episode": episode,
                }
                self.save_logs_and_checkpoints(
                    global_step=global_step,
                    logs_dict=status,
                    client_states=client_states,
                )

                global_step += (
                    self.max_epochs
                    * (self.vllm_generate_batch_size * self.n_samples_per_prompt)
                    // self.train_batch_size
                )

                if global_step >= self.max_steps:
                    break

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
