# infer.mk - batch inference, guardrail evaluation, and GPT-4o-judge targets.
# Whenever an evaluation needs to trigger a guardrail or LLM-as-a-judge, we
# run inference once to generate responses, then again to compute the metric.

INFER = ++deepspeed.zero_stage=0 ++data.use_eval=false
OAI_JUDGE = ++jsonl_file_name=$* ++gpt_model_name="gpt-4o"

# the first argument is dataset name
define INFER_ARGS
infer=default \
	$(1) \
	++infer.prompt_max_len=$(2) ++infer.generate_max_len=$(3) \
	++infer.compute_metric=$(4) \
	++model.generation_args.temperature=$(TEMP02)
endef

# 1 guardrail model 2 dataset name
define GUARD_ARGS
infer=guardrail \
	$(1) \
	$(2) \
	++infer.compute_metric=$(3) \
	++infer.prompt_max_len=$(4) \
	++infer.generation_filename=$(5) \
	++save_dataset=$(6) \
	++model.generation.temperature=$(TEMP02) \
	++data.use_response=true \
	++data.use_prefill=false
endef

# general dataset inference args
gsm8shot_args            := $(call INFER_ARGS,$(gsm_8shot),3000,3000,true)
gsm0shot_args            := $(call INFER_ARGS,$(gsm_0shot),1000,3000,true)
math500_args             := $(call INFER_ARGS,$(math500),1000,7000,true)
math500dsqwen1b_args     := $(call INFER_ARGS,$(math500dsqwen1b),8000,7000,true)
math500dsqwen32b_args    := $(call INFER_ARGS,$(math500dsqwen32b),8000,7000,true)
strongrej_args           := $(call INFER_ARGS,$(strongrej),1000,3000,false)
strongrejdsqwen1b_args   := $(call INFER_ARGS,$(strongrejdsqwen1b),4000,3000,false)
strongrejdsqwen32b_args  := $(call INFER_ARGS,$(strongrejdsqwen32b),4000,3000,false)
strongrejdsqwen1bfull_args := $(call INFER_ARGS,$(strongrejdsqwen1bfull),4000,3000,false)
fortress_args            := $(call INFER_ARGS,$(fortress),3000,3000,false)
hexphi_args              := $(call INFER_ARGS,$(hexphi),1000,3000,false)
advbench_args            := $(call INFER_ARGS,$(advbench),1000,3000,false)
advbenchpredsqwen1b_args := $(call INFER_ARGS,$(advbenchpredsqwen1b),1500,3000,false)
aime24_args              := $(call INFER_ARGS,$(aime24),1000,23000,true) ++infer.pass_k=16 ++model.generation_args.temperature=$(TEMP06)
harmchain_args           := $(call INFER_ARGS,$(harmchain),1000,3000,false)
harmchainprefill_args    := $(call INFER_ARGS,$(harmchainprefill),1000,3000,false)
wjbharm_args             := $(call INFER_ARGS,$(wjb_eval_advharm),1000,3000,false)
wjbbenign_args           := $(call INFER_ARGS,$(wjb_train_advbenign),1000,3000,false)   
star1k_args              := $(call INFER_ARGS,$(star1k),1000,3000,false)
staror1k_args            := $(call INFER_ARGS,$(staror1k),1000,3000,false)
xstest_args              := $(call INFER_ARGS,$(xstest),1000,3000,false)
fortressor_args          := $(call INFER_ARGS,$(fortressor),1000,3000,false)

gsm8k0shotprefill_args = $(call INFER_ARGS,$(gsm8k_0shot_prefill),3000,true,32) # TODO
hexphipoison_args    = $(call INFER_ARGS,$(hexphipoison),500,false,42)
advbenchprefill_args = $(call INFER_ARGS,$(advbenchprefill),1500,false,35)
advbenchpreds_args = $(call INFER_ARGS,$(advbenchprefillds),1500,false,35)
advbenchprefilldsllama70b_args = $(call INFER_ARGS,$(advbenchprefilldsllama70b),1500,false,35)
advbenchpoison_args  = $(call INFER_ARGS,$(advbenchpoison),500,false,35)

# LLM specific datasets
# llama31-specific dataset settings
mmlu_args_llama31_8b := $(call INFER_ARGS,$(mmlu_llama31_8b),3500,50,true)
arcc_args_llama31_8b := $(call INFER_ARGS,$(arcc_llama31_8b),500,50,true)

# llama32-specific dataset settings
mmlu_args_llama32_1b := $(call INFER_ARGS,$(mmlu_llama32_1b),3500,50,true)
arcc_args_llama32_1b := $(call INFER_ARGS,$(arcc_llama32_1b),500,50,true)


# guardrail-specific settings for training and evaluation, ensure prompt len is total length of infer + 200 for guardrail template
# 	1. if training for RS, infer.generation_filename is null, and save_dataset is true
# 	2. if evaluating for safety score, infer.generation_filename is $*, and save_dataset is false
# evaluate from model generation, we dont support llamaguard3_1b for now due to the conversation format mismatch
INFER_llamaguard3_8b_strongrej           = $(call GUARD_ARGS,$(llamaguard3_8b),$(strongrej),true,4200,$*,false)
INFER_llamaguard3_8b_fortress            = $(call GUARD_ARGS,$(llamaguard3_8b),$(fortress),true,6200,$*,false)
INFER_llamaguard3_8b_advbench            = $(call GUARD_ARGS,$(llamaguard3_8b),$(advbench),true,4200,$*,false)
INFER_llamaguard3_8b_harmchain           = $(call GUARD_ARGS,$(llamaguard3_8b),$(harmchain),true,4200,$*,false)
INFER_llamaguard3_8b_harmchainprefill    = $(call GUARD_ARGS,$(llamaguard3_8b),$(harmchainprefill),true,4200,$*,false)
INFER_llamaguard3_8b_wjbharm             = $(call GUARD_ARGS,$(llamaguard3_8b),$(wjb_eval_advharm),true,4100,$*,false)

INFER_granite_guardian31_2b_strongrej    = $(call GUARD_ARGS,$(granite_guardian31_2b),$(strongrej),true,4200,$*,false)
INFER_granite_guardian31_2b_fortress     = $(call GUARD_ARGS,$(granite_guardian31_2b),$(fortress),true,6200,$*,false)

INFER_granite_guardian31_8b_strongrej    = $(call GUARD_ARGS,$(granite_guardian31_8b),$(strongrej),true,4200,$*,false)
INFER_granite_guardian31_8b_fortress     = $(call GUARD_ARGS,$(granite_guardian31_8b),$(fortress),true,6200,$*,false)

INFER_wildguard_7b_staror1k              = $(call GUARD_ARGS,$(wilguard_7b),$(staror1k),true,4100,$*,false)
INFER_wildguard_7b_xstest                = $(call GUARD_ARGS,$(wilguard_7b),$(xstest),true,4100,$*,false)
INFER_wildguard_7b_fortressor            = $(call GUARD_ARGS,$(wilguard_7b),$(fortressor),true,4100,$*,false)
INFER_wildguard_7b_wjbbenign             = $(call GUARD_ARGS,$(wilguard_7b),$(wjb_train_advbenign),true,4100,$*,false)

INFER_llamaguard3_8b_advbenchpredsqwen1b   = $(call GUARD_ARGS,$(llamaguard3_8b),$(advbenchpredsqwen1b),true,4700,$*,false)

INFER_llamaguard3_8b_harmchain                = $(call GUARD_ARGS,$(llamaguard3_8b),$(harmchain),true,4020,null,false)
INFER_granite_guardian31_8b_harmchain         = $(call GUARD_ARGS,$(granite_guardian31_8b),$(harmchain),true,4020,null,false)

#
# Auto parsing INFER_<model>_<dataset>, here model is LLM-only, not guardrail
#

# Fallback macro logic
define GET_DATASET_ARGS
$(if $(value $(2)_args_$(call FAMILY_PREFIX,$(1))), \
  $($(2)_args_$(call FAMILY_PREFIX,$(1))), \
  $($(2)_args))
endef

define INFER_ENTRY
INFER_$(1)_$(2) := $(call RESOLVE_MODEL_NAME,$(1))$(call GET_DATASET_ARGS,$(1),$(2))
endef

define INFER_BLOCK
$(foreach m,$(1), \
  $(eval $(call DEFINE_MODEL_FROM_RESOLVER,$(m))) \
  $(foreach d,$(2), \
    $(eval $(call INFER_ENTRY,$(m),$(d))) \
  ) \
)
endef

# we support both hf generate with deepspeed and vllm generate with python in all batch inferences
experiment/infer/%/.done_infer:
> @echo "[INFO] Running inference on: $*"
> @mkdir -p $(dir $@)
> $(PYTHON) -m src.infer_vllm ++name=$* $(INFER) $(INFER_$*) 

# guardrail evaluating the safety of model generations
experiment/infer/%/.done_infer_guardrail:
> @echo "[INFO] Running guardrail-based evaluation for: $*"
> @mkdir -p $(dir $@)
> $(PYTHON) -m src.infer_vllm ++name=$* $(INFER) $(INFER_$*)
> $(PYTHON) -m src.infer_vllm ++name=$* $(INFER) $(INFER_$(guardrail)_$(lastword $(subst _, ,$*)))

experiment/infer/%/.done_infer_oai_safety_judge:
> @echo "[INFO] Running oai-as-a-judge safety evaluation for: $*"
> @mkdir -p $(dir $@)  # Ensure the directory exists
> $(PYTHON) -m src.infer_vllm ++name=$* $(INFER) $(INFER_$*)
> $(PYTHON) -m src.oai_judge ++name=$* $(OAI_JUDGE) ++judge_template="safety_judge.txt"

experiment/infer/%/.done_infer_oai_or_judge:
> @echo "[INFO] Running oai-as-a-judge overrefusal evaluation for: $*"
> @mkdir -p $(dir $@)  # Ensure the directory exists
> $(PYTHON) -m src.infer_vllm ++name=$* $(INFER) $(INFER_$*)
> $(PYTHON) -m src.oai_judge ++name=$* $(OAI_JUDGE) ++judge_template="overrefusal_judge.txt"

