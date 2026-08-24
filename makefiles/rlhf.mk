# rlhf.mk - RLHF / GRPO / DAPO run targets (RECAP training): per-node GPU
# configs, the shared DAPO-style GRPO recipe, and GRPO_<model>_<dataset>
# variables consumed by the generic experiment/grpo/%/.done_grpo recipe.
# Also includes the ray-up helper for launching a multi-node Ray cluster.

ray-up:
> @echo "[INFO] Gathering node list from SLURM_JOB_NODELIST"
> @hostnames=($$(scontrol show hostnames $$SLURM_JOB_NODELIST)); \
> head_node=$${hostnames[0]}; \
> head_ip=$$(getent hosts $$head_node | awk '{ print $$1 }'); \
> port=9988; \
> echo "[INFO] Head node: $$head_node ($$head_ip:$$port)"; \
> echo "[INFO] All nodes: $${hostnames[*]}"; \
> echo "[INFO] Launching Ray head on $$head_node..."; \
> srun --nodes=1 --ntasks=1 -w $$head_node bash -c "\
> 	source ~/miniconda3/etc/profile.d/conda.sh && \
> 	conda activate llm && \
> 	ray stop && \
> 	ray start --head --port=$$port --block" & \
> for node in $${hostnames[@]:1}; do \
> 	echo "[INFO] Launching Ray worker on $$node..."; \
> 	srun --nodes=1 --ntasks=1 -w $$node bash -c "\
> 		source ~/miniconda3/etc/profile.d/conda.sh && \
> 		conda activate llm && \
> 		ray stop && \
> 		ray start --address=$$head_ip:$$port --block" & \
> done; \
> wait

# GPU configs for GRPO
GRPO_1NODE := ++train.micro_train_batch_size=1 ++train.train_batch_size=4 \
	++train.policy_num_gpus=4 ++train.ref_num_gpus=0 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=1 \
	++train.vllm_num_engines=3 ++train.vllm_tensor_parallel_size=1 \
	++train.vllm_generate_batch_size=1 ++train.n_samples_per_prompt=16 ++train.micro_rollout_batch_size=4

GRPO_2NODES := ++train.micro_train_batch_size=1 ++train.train_batch_size=8 \
	++train.policy_num_gpus=8 ++train.ref_num_gpus=1 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=1 \
	++train.vllm_num_engines=6 ++train.vllm_tensor_parallel_size=1 \
	++train.vllm_generate_batch_size=2 ++train.n_samples_per_prompt=20 ++train.micro_rollout_batch_size=5

GRPO_3NODES := ++train.vllm_num_engines=8 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=12 ++train.ref_num_gpus=2 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=2 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=12 \
	++train.vllm_generate_batch_size=3 ++train.n_samples_per_prompt=8 ++train.micro_rollout_batch_size=2

GRPO_4NODES := ++train.vllm_num_engines=14 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=16 ++train.ref_num_gpus=0 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=2 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=16 \
	++train.vllm_generate_batch_size=4 ++train.n_samples_per_prompt=16 ++train.micro_rollout_batch_size=4

GRPO_5NODES := ++train.vllm_num_engines=16 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=20 ++train.ref_num_gpus=2 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=2 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=20 \
	++train.vllm_generate_batch_size=4 ++train.n_samples_per_prompt=15 ++train.micro_rollout_batch_size=3

GRPO_6NODES := ++train.vllm_num_engines=20 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=24 ++train.ref_num_gpus=2 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=2 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=24 \
	++train.vllm_generate_batch_size=6 ++train.n_samples_per_prompt=16 ++train.micro_rollout_batch_size=4

GRPO_8NODES := ++train.vllm_num_engines=30 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=32 ++train.ref_num_gpus=0 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=2 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=32 \
	++train.vllm_generate_batch_size=8 ++train.n_samples_per_prompt=16 ++train.micro_rollout_batch_size=4

GRPO_12NODES := ++train.vllm_num_engines=44 ++train.vllm_tensor_parallel_size=1 \
	++train.policy_num_gpus=48 ++train.ref_num_gpus=0 ++train.forward_reward_num_gpus=0 ++train.generative_reward_num_gpus=4 \
	++train.micro_train_batch_size=1 ++train.train_batch_size=48 \
	++train.vllm_generate_batch_size=12 ++train.n_samples_per_prompt=16 ++train.micro_rollout_batch_size=4

# Reward Model
RM_granite_guardian31_8b := ++reward_model.guardrail_binary.name='ibm-granite/granite-guardian-3.1-8b'

# ==================== GRPO ====================
# This recipe is effectively DAPO (paper App. C): decoupled "clip-higher"
# eps=[0.2,0.28], no KL penalty, dynamic sampling to drop all-good/all-bad
# rollout groups, and a soft overlong buffer. RECAP reuses this recipe UNCHANGED
# for both prefilled and standard prompts -- the only difference is that the
# dataset (harmchain*/staror*) carries counter-aligned CoT prefills.
# Target naming: GRPO_<model>_<dataset> composes <nodes-GPU-config> <model>
# <dataset> [+ optional reward-model override].
GRPO := ++task=grpo ++model.policy_model.generation_args.temperature=1.0 ++train.vllm_gpu_memory_utilization=0.95 \
	++train.generate_max_len=7000 ++train.overlong_buffer.enable=true ++train.overlong_buffer.len=500 \
	++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] \
	++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,0.98] \
	++train.max_training_steps=-1 ++train.num_episodes=1

# GRPO_<exp> = <nodes> <model> <dataset> <optional changes on RM>

# prefilling dynamics: harmful + overrefusal
GRPO_dsllama_8b_harmchainstaror             = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainstaror) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainr025staror         = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainr025staror) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainpre700staror       = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainpre700staror) $(RM_granite_guardian31_8b)

GRPO_dsllama_8b_harmchainwjbadvbenign       = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainwjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainpre100wjbadvbenign = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainpre100wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainpre300wjbadvbenign = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainpre300wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainpre500wjbadvbenign = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainpre500wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainpre700wjbadvbenign = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainpre700wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainr025wjbadvbenign   = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainr025wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainr050wjbadvbenign   = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainr050wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainr075wjbadvbenign   = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainr075wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainr100wjbadvbenign   = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainr100wjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainwjbadvbenignpre    = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainwjbadvbenignpre) $(RM_granite_guardian31_8b)

GRPO_dsllama_8b_harmchainstarorpre          = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainstarorpre) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_harmchainprestarorpre       = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(harmchainprestarorpre) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_starstaror                  = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(starstaror) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_starprestaror               = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(starprestaror) $(RM_granite_guardian31_8b)

GRPO_dsqwen_7b_harmchainstaror              = $(GRPO_4NODES) $(rlhf_dsqwen_7b) $(harmchainstaror) $(RM_granite_guardian31_8b)
GRPO_dsqwen_7b_harmchainprestarorpre        = $(GRPO_4NODES) $(rlhf_dsqwen_7b) $(harmchainprestarorpre) $(RM_granite_guardian31_8b)

# math + harmful + overrefusal
GRPO_dsllama_8b_mathgsmharmchainwjbadvbenign = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(mathgsmharmchainwjbadvbenign) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_mathgsmharmchainprewjbadvbenignpre = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(mathgsmharmchainprewjbadvbenignpre) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_mathgsmharmchainstaror       = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(mathgsmharmchainstaror) $(RM_granite_guardian31_8b)
GRPO_dsllama_8b_mathgsmharmchainprestarorpre = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(mathgsmharmchainprestarorpre) $(RM_granite_guardian31_8b)
GRPO_dsqwen_14b_mathgsmharmchainstaror       = $(GRPO_8NODES) $(rlhf_dsqwen_14b) $(mathgsmharmchainstaror) $(RM_granite_guardian31_8b) ++train.generate_max_len=4000
GRPO_dsqwen_14b_mathgsmharmchainprestarorpre = $(GRPO_8NODES) $(rlhf_dsqwen_14b) $(mathgsmharmchainprestarorpre) $(RM_granite_guardian31_8b) ++train.generate_max_len=4000

GRPO_dsllama_8b_mathgsm                      = $(GRPO_4NODES) $(rlhf_dsllama_8b) $(mathgsm)

GRPO_dsllama_8b_staror = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(staror)
GRPO_dsllama_8b_starorpre300 = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(starorpre300)

GRPO_dsllama_8b_stargsmmathstaror = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(stargsmmathstaror) $(RM)
GRPO_dsllama_8b_starpre300gsmmathstarorpre300 = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(starpre300gsmmathstarorpre300) $(RM)

GRPO_dsllama_8b_starpre300staror = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(starpre300staror) $(RM)

GRPO_dsllama_8b_harmchainprecot600staror = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(harmchainprecot600staror) $(RM)

GRPO_dsllama_8b_stargsmmath = $(GRPO_recipe) $(rlhf_dsllama_8b) $(GRPO_4NODES) $(stargsmmath) ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b'


GRPO_dsqwen_14b_stargsmmath  := $(GRPO_recipe) $(rlhf_dsqwen_14b) $(GRPO_8NODES) $(stargsmmath) ++train.generate_max_len=4000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b'

# GRPO_xx: model, dataset, gpu config, perhaps additional config
GRPO_dsqwen_7b_purebad = $(rlhf_dsqwen_7b) $(GRPO_1NODE) $(purebad) ++train.generate_max_len=3000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++data.d100.reward_name=guardrail_logit ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,1.] ++train.max_training_steps=50 ++train.num_episodes=2
GRPO_dsllama_8b_purebad = $(rlhf_dsllama_8b) $(purebad) $(GRPO_2NODES) ++train.generate_max_len=4000
GRPO_dsqwen_14b_purebad = $(rlhf_dsqwen_14b) $(purebad) $(GRPO_4NODES)
GRPO_dsqwen_32b_purebad = $(GRPO_4NODES) $(rlhf_dsqwen_32b) $(purebad)

TMP := ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++data.d2.reward_name=guardrail_logit
TMP_UNSAFE := ++reward_model.guardrail_logit_unsafe.name='ibm-granite/granite-guardian-3.1-8b' ++data.d1.reward_name=guardrail_logit_unsafe

GRPO_dsllama_8b_star1unsafe = $(rlhf_dsllama_8b) $(GRPO_5NODES) $(star1) ++train.generate_max_len=5000 $(TMP_UNSAFE)
GRPO_dsllama_8b_harmchain1k_sft_star1unsafe = $(rlhf_dsllama_8b_harmchain1k_sft) $(GRPO_5NODES) $(star1) ++train.generate_max_len=5000 $(TMP_UNSAFE)
GRPO_dsllama_8b_harmchain1k_sft_star1unsafenokl = $(rlhf_dsllama_8b_harmchain1k_sft) $(GRPO_5NODES) $(star1) ++train.generate_max_len=5000 $(TMP_UNSAFE) ++train.init_kl_coef=0 ++train.use_kl_loss=false

GRPO_dsllama_8b_harmchain = $(rlhf_dsllama_8b) $(GRPO_2NODES) $(harmchain) ++train.generate_max_len=5000 $(TMP)
GRPO_dsllama_8b_harmchainprefill = $(rlhf_dsllama_8b) $(GRPO_2NODES) $(harmchainprefill) ++train.generate_max_len=5000 $(TMP)

GRPO_dsqwen_7b_gsm1kharmchain = $(rlhf_dsqwen_7b) $(GRPO_3NODES) $(gsm1kharmchain) ++train.generate_max_len=5000 $(TMP)
GRPO_dsqwen_7b_gsm1kharmchainprefill = $(rlhf_dsqwen_7b) $(GRPO_3NODES) $(gsm1kharmchainprefill) ++train.generate_max_len=5000 $(TMP)
GRPO_dsllama_8b_gsm1kharmchain = $(rlhf_dsllama_8b) $(GRPO_3NODES) $(gsm1kharmchain) ++train.generate_max_len=5000 $(TMP)
GRPO_dsllama_8b_gsm1kharmchainprefill = $(rlhf_dsllama_8b) $(GRPO_3NODES) $(gsm1kharmchainprefill) ++train.generate_max_len=5000 $(TMP)

GRPO_dsllama_8b_dsmath = $(rlhf_dsllama_8b) $(GRPO_4NODES) $(dsmath) ++train.generate_max_len=7000 ++train.overlong_buffer.enable=true ++train.overlong_buffer.len=500 ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,1.] ++train.max_training_steps=-1 ++train.num_episodes=1 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup
GRPO_dsllama_8b_dsmathlen = $(rlhf_dsllama_8b) $(GRPO_4NODES) $(dsmath) ++train.generate_max_len=6000 ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,1.] ++train.max_training_steps=-1 ++train.num_episodes=1 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup

GRPO_dsllama_8b_math8k = $(rlhf_dsllama_8b) $(GRPO_4NODES) $(math8k) ++train.generate_max_len=7000 ++train.overlong_buffer.enable=true ++train.overlong_buffer.len=500 ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,1.] ++train.max_training_steps=-1 ++train.num_episodes=1 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup

GRPO_dsllama_8b_gsm1kharmchainnoklcliphighdynamic    = $(rlhf_dsllama_8b) $(GRPO_4NODES) $(gsm1kharmchain) ++train.generate_max_len=5000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,0.97] ++train.max_training_steps=400 ++train.num_episodes=2 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup
GRPO_dsllama_8b_gsm1kharmchainprenoklcliphighdynamic = $(rlhf_dsllama_8b) $(GRPO_4NODES) $(gsm1kharmchainprecot600) ++train.generate_max_len=5000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,0.97] ++train.max_training_steps=400 ++train.num_episodes=2 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup

GRPO_dsqwen_14b_harmchain = $(GRPO_4NODES) $(rlhf_dsqwen_14b) $(harmchain) ++train.generate_max_len=5000
GRPO_dsqwen_14b_harmchainprefill = $(GRPO_4NODES) $(rlhf_dsqwen_14b) $(harmchainprefill) ++train.generate_max_len=5000

GRPO_dsqwen_14b_gsm1kharmchain    := $(rlhf_dsqwen_14b) $(GRPO_12NODES) $(gsm1kharmchain) ++train.generate_max_len=5000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,0.98] ++train.max_training_steps=400 ++train.num_episodes=1 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup
GRPO_dsqwen_14b_gsm1kharmchainpre := $(rlhf_dsqwen_14b) $(GRPO_8NODES) $(gsm1kharmchainprecot600) ++train.generate_max_len=5000 ++reward_model.guardrail_logit.name='ibm-granite/granite-guardian-3.1-8b' ++train.init_kl_coef=0 ++train.use_kl_loss=false ++train.ratio_clip_eps=[0.2,0.28] ++train.dynamic_sampling=true ++train.dynamic_sampling_reward_range=[0.,0.98] ++train.max_training_steps=400 ++train.num_episodes=1 ++train.policy_learning_rate=1e-6 ++train.policy_lr_scheduler=constant_with_warmup


experiment/grpo/%/.done_grpo:
> @echo "[INFO] Running GRPO experiment: $*_grpo"
> @mkdir -p $(patsubst %/,%, $(dir $@))_grpo
> $(PYTHON) -m src.rlhf ++name=$*_grpo $(GRPO) $(GRPO_$*)
