# sft.mk - supervised fine-tuning run targets: SFT_<model>_<dataset> variables
# compose a model (model.mk) with a dataset (dataset.mk) into Hydra overrides
# consumed by the generic experiment/sft/%/.done_sft recipe below.

# prompt 1000, response 4000 suits for most training, increase or decrease if needed
SFT := ++data.use_response=true ++data.use_prefill=false ++train.prompt_max_len=1000 ++train.generate_max_len=4000

SFT_1NODE  := ++train.micro_train_batch_size=1 ++train.train_batch_size=8 
SFT_2NODES := ++train.micro_train_batch_size=1 ++train.train_batch_size=16
SFT_4NODES := ++train.micro_train_batch_size=1 ++train.train_batch_size=32

# ================== non-reasoning models ==================
SFT_llama3_8b_purebad = $(llama3_8b) $(purebad)
SFT_llama31_8b_gsm8k = $(llama31_8b) $(gsm8k_train)
SFT_llama31_8b_purebad = $(llama31_8b) $(purebad)
SFT_llama32_1b_gsm8k = $(llama32_1b) $(gsm8k_train)
SFT_llama32_1b_purebad = $(llama32_1b) $(purebad)
SFT_llama32_1b_purebadsfx1 = $(llama32_1b) $(purebadsfx1)
SFT_llama32_1b_purebadpfx1 = $(llama32_1b) $(purebadpfx1)
SFT_llama32_1b_beavertails = $(llama32_1b) $(beavertails)
SFT_llama32_1b_hhrlhf = $(llama32_1b) $(hhrlhf)
SFT_llama32_1b_gsm8kpurebad = $(llama32_1b) $(gsm8kpurebad)
SFT_llama32_1b_gsm8kpurebadsfx1 = $(llama32_1b) $(gsm8kpurebadsfx1)
SFT_llama33_70b_gsm8k = $(llama33_70b) $(gsm8k_train)

SFT_mistralsmall_24b_purebad = $(mistral_small_24b) $(purebad)
SFT_llama2_7b_purebad = $(llama2_7b) $(purebad)
SFT_gemma3_1b_purebad = $(gemma3_1b) $(purebad)
SFT_granite33_2b_purebad = $(granite33_2b) $(purebad)
SFT_qwen25_3b_purebad = $(qwen25_3b) $(purebad)

# ================== reasoning models ==================
SFT_dsllama_8b_harmchain1k = $(SFT_4NODES) $(dsllama_8b) $(harmchain1ksft)
SFT_dsqwen_14b_harmchain1k = $(SFT_4NODES) $(dsqwen_14b) $(harmchain1ksft) ++train.max_epochs=10
SFT_dsqwen_32b_harmchain1k = $(SFT_4NODES) $(dsqwen_32b) $(harmchain1ksft) ++train.max_epochs=10 ++train.generate_max_len=3000
SFT_dsqwen_14b_safechain   = $(SFT_4NODES) $(dsqwen_14b) $(safechain) ++train.max_epochs=2 ++train.generate_max_len=3000

# mixture of safe + x
SFT_llama32_1b_purebadpuresafe = $(llama32_1b) $(purebadpuresafe)
SFT_llama32_1b_puresafepurebadorder = $(llama32_1b) $(puresafepurebadorder)
SFT_llama32_1b_purebadpoisonpuresafe = $(llama32_1b) $(purebadpoisonpuresafe)
SFT_llama32_1b_gsm8kpurebadpuresafe100 = $(llama32_1b) $(gsm8kpurebadpuresafe100)
SFT_llama32_1b_gsm8kpurebadpuresafeorder = $(llama32_1b) $(gsm8kpurebadpuresafeorder)

experiment/sft/%/.done_sft:
> @echo "[INFO] Running SFT experiment: $*_sft"
> @mkdir -p $(patsubst %/,%, $(dir $@))_sft
> @if [ -n "$$SLURM_JOB_NODELIST" ]; then \
> 	$(MAKE) gen_hostfile; \
> fi
> $(DEEPSPEED) --module src.sft ++name=$*_sft $(SFT) $(SFT_$*)
