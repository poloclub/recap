# model.mk - model registry: HF model ids for policy/base models, guardrail
# and overrefusal-judge models, plus the +model@... overrides used to wire
# policy/ref models into rlhf and to resolve trained (local) checkpoint paths.

# root
gpt2    := model=gpt2
llama   := model=llama
gemma   := model=gemma
mistral := model=mistral
qwen    := model=qwen
qwen3_thinking := model=qwen3_thinking
dsr1    := model=deepseekr1
granite := model=granite
granite_guardian := model=granite_guardian
wildguard := model=wildguard
phi      := model=phi

# non-reasoning models
llama2_7b    := $(llama) ++model.name="meta-llama/Llama-2-7b-chat-hf"
llama2_13b   := $(llama) ++model.name="meta-llama/Llama-2-13b-chat-hf"
llama3_8b    := $(llama) ++model.name="meta-llama/Meta-Llama-3-8B-Instruct"
llama3_70b   := $(llama) ++model.name="meta-llama/Meta-Llama-3-70B-Instruct"
llama31_8b   := $(llama) ++model.name="meta-llama/Llama-3.1-8B-Instruct"
llama31_70b  := $(llama) ++model.name="meta-llama/Llama-3.1-70B-Instruct"
llama32_1b   := $(llama) ++model.name="meta-llama/Llama-3.2-1B-Instruct"
llama32_3b   := $(llama) ++model.name="meta-llama/Llama-3.2-3B-Instruct"
llama33_70b  := $(llama) ++model.name="meta-llama/Llama-3.3-70B-Instruct"
gemma3_1b    := $(gemma) ++model.name="google/gemma-3-1b-it"
gemma3_4b    := $(gemma) ++model.name="google/gemma-3-4b-it" # cannot run
qwen25_3b    := $(qwen) ++model.name="Qwen/Qwen2.5-3B-Instruct"
granite33_2b := $(granite) ++model.name="ibm-granite/granite-3.3-2b-instruct"
mistral_24b  := $(mistral) ++model.name="mistralai/Mistral-Small-24B-Instruct-2501"

# reasoning models
dsqwen_1b    := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dsqwen_7b    := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

dsllama_8b   := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

dsqwen_14b   := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
dsqwen_32b   := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
dsllama_70b  := $(dsr1) ++model.name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
starllama_8b := $(dsr1) ++model.name="UCSC-VLAA/STAR1-R1-Distill-8B"
starqwen_14b := $(dsr1) ++model.name="UCSC-VLAA/STAR1-R1-Distill-14B"
safechainllama_8b := $(dsr1) ++model.name="UWNSL/DeepSeek-R1-Distill-Llama-8B-SafeChain"

phi4_14b     := $(phi) ++model.name="microsoft/Phi-4-reasoning"
phi4plus_14b := $(phi) ++model.name="microsoft/Phi-4-reasoning-plus"

qwen3think_4b := $(qwen3_thinking) ++model.name="Qwen/Qwen3-4B-Thinking-2507"

# define rlhf policy reference model
rlhf_ref_model := ++model.ref_model.template_args=null ++model.ref_model.lora=null ++model.ref_model.chat_template=null

rlhf_dsqwen_7b = +model@model.policy_model=deepseekr1 ++model.policy_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
	+model@model.ref_model=deepseekr1 ++model.ref_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" $(rlhf_ref_model)
rlhf_dsllama_8b = +model@model.policy_model=deepseekr1 ++model.policy_model.name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
	+model@model.ref_model=deepseekr1 ++model.ref_model.name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" $(rlhf_ref_model)
rlhf_dsqwen_14b = +model@model.policy_model=deepseekr1 ++model.policy_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
	+model@model.ref_model=deepseekr1 ++model.ref_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" $(rlhf_ref_model)
rlhf_dsqwen_32b = +model@model.policy_model=deepseekr1 ++model.policy_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
	+model@model.ref_model=deepseekr1 ++model.ref_model.name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" $(rlhf_ref_model)

rlhf_llama32_1b = +model@model.policy_model=llama ++model.policy_model.name="meta-llama/Llama-3.2-1B-Instruct" \
	+model@model.ref_model=llama ++model.ref_model.name="meta-llama/Llama-3.2-1B-Instruct" $(rlhf_ref_model)

rlhf_dsllama_8b_harmchain1k_sft = +model@model.policy_model=deepseekr1 ++model.policy_model.name=$(call SET_MODEL_PATH,dsllama_8b_harmchain1k_sft) \
	+model@model.ref_model=deepseekr1 ++model.ref_model.name=$(call SET_MODEL_PATH,dsllama_8b_harmchain1k_sft) $(rlhf_ref_model)

# guardrail && overrefusal models
llamaguard3_1b        := $(llama) ++model.name="meta-llama/Llama-Guard-3-1B"
llamaguard3_8b        := $(llama) ++model.name="meta-llama/Llama-Guard-3-8B"
granite_guardian3_8b  := $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.0-8b"
granite_guardian31_2b := $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.1-2b"
granite_guardian31_8b := $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.1-8b"
granite_guardian32_5b := $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.2-5b"
wilguard_7b           := $(wildguard) ++model.name="allenai/wildguard"

# reference model for value sft, perhaps we will move it to config yaml later
ref_llama2_7b  = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-2-7b-chat-hf"
ref_llama32_1b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.2-1B-Instruct"
ref_llama32_3b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.2-3B-Instruct"
ref_llama31_8b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.1-8B-Instruct"
ref_gemma3_1b = +model@ref_model=gemma ++ref_model.name="google/gemma-3-1b-it"
ref_gemma3_4b = +model@ref_model=gemma ++ref_model.name="google/gemma-3-4b-it"
ref_granite33_2b = +model@ref_model=granite ++ref_model.name="ibm-granite/granite-3.3-2b-instruct"
ref_qwen25_3b = +model@ref_model=qwen ++ref_model.name="Qwen/Qwen2.5-3B-Instruct"

# sft model weights
define SET_MODEL_PATH
../../$(lastword $(subst _, ,$(1)))/$(1)/final_hf
endef

# trained model weights, currently we only support sft 
# split `llama32_1b_purebad` → `llama32_1b` all model name needs to follow this pattern, for OSS model, name_size, for local model name_size_dataset
FAMILY_PREFIX = $(word 1,$(subst _, ,$(1)))_$(word 2,$(subst _, ,$(1)))

define RESOLVE_MODEL_NAME
$(if $(filter $(1),$(call FAMILY_PREFIX,$(1))),\
$(call REQUIRE_DEFINED,$(1))$($(1)), \
$(call REQUIRE_DEFINED,$(call FAMILY_PREFIX,$(1)))$($(call FAMILY_PREFIX,$(1))) ++model.name=$(call SET_MODEL_PATH,$(1)))
endef

define DEFINE_MODEL_FROM_RESOLVER
$(eval $(1) := $(call RESOLVE_MODEL_NAME,$(1)))
endef

# dpo model weights
dpo_folder = ../../dpo/$*/final_hf # experiment/dpo/$*/final_hf
llama32_1b_openrlhf_mixture2 = $(llama32) ++model.name=$(dpo_folder)

