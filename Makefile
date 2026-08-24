# ========================== Makefile Structure ==========================
# Top-level entrypoints (venv, sft, rlhf/grpo, infer, guard, chat, ...) that
# assemble Hydra CLI overrides and dispatch to experiment/<stage>/<name>
# targets. Includes makefiles/{misc,model,dataset,sft,rlhf,infer}.mk and
# CONFIG.mk (the experiment matrix).


SHELL := /bin/bash
VENV_NAME := llm
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE := eval "$$(conda shell.bash hook)" && conda activate && conda activate $(VENV_NAME)
PIP := $(CONDA_ACTIVATE) && pip3
PYTHON := python

# Dynamically find an available port and launch deepspeed
FREE_PORT := $(shell python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
ACCELERATE := $(CONDA_ACTIVATE) && accelerate launch
# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# check if a variable is defined or not
define REQUIRE_DEFINED
$(if $(value $(1)),,$(error [ERROR] Variable '$(1)' is not defined))
endef


# Top-level Makefile
MKDIR := makefiles
include $(MKDIR)/misc.mk
include $(MKDIR)/model.mk
include $(MKDIR)/dataset.mk
include $(MKDIR)/sft.mk
include $(MKDIR)/rlhf.mk
include $(MKDIR)/infer.mk
include CONFIG.mk


# ==================== Environment setup ====================
# Prebuilt flash-attn wheel for CUDA 12 / torch 2.6 / python 3.10. Swap this for
# the wheel matching your system: https://github.com/Dao-AILab/flash-attention/releases
FLASH_ATTN_WHEEL := https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Create the `$(VENV_NAME)` conda env and install all dependencies + this package.
# torch and flash-attn are installed first so openrlhf/vllm resolve against them
# (and flash-attn does not get built from source).
venv:
> conda create -y -n $(VENV_NAME) python=3.10
> $(CONDA_ACTIVATE) && pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
> $(CONDA_ACTIVATE) && pip install --no-cache-dir $(FLASH_ATTN_WHEEL)
> $(CONDA_ACTIVATE) && pip install --no-cache-dir -r requirements.txt
> $(CONDA_ACTIVATE) && pip install -e .
> @echo "[INFO] conda env '$(VENV_NAME)' is ready. Activate with: conda activate $(VENV_NAME)"

.PHONY: venv


.SECONDARY:

# CLI-friendly entrypoint
default: help

# auto generate hostfile for multi-node deepspeed, assuming in slurm mode
HOSTFILE := $(HOME)/.cache/hostfile
gen_hostfile:
> @if [ "$$SLURM_NODEID" = "0" ]; then \
>   echo "Generating hostfile on main node..."; \
>   rm -f $(HOSTFILE); \
>   scontrol show hostnames $$SLURM_JOB_NODELIST | awk '{print $$1, "slots=8"}' > $(HOSTFILE); \
>   cat $(HOSTFILE); \
> else \
>   echo "Waiting for hostfile on node $$SLURM_NODEID..."; \
>	  while [ ! -f $(HOSTFILE) ]; do sleep 1; done; \
>	  echo "Hostfile detected"; \
> fi

# used for meta aws slurm and ssh interactive
ifdef SLURM_JOB_NODELIST
DEEPSPEED = deepspeed --hostfile=$(HOSTFILE) --no_ssh --master_addr=$(shell scontrol show hostnames $(SLURM_JOB_NODELIST) | head -n 1) --node_rank=${SLURM_NODEID} --master_port=30000
else
DEEPSPEED = deepspeed --master_port=30000
endif

# batch inference
infer: experiment/infer/$(name)/.done_infer
guard: experiment/infer/$(name)/.done_infer_guardrail
oai_safety_judge: experiment/infer/$(name)/.done_infer_oai_safety_judge
oai_or_judge: experiment/infer/$(name)/.done_infer_oai_or_judge

# supervised fine-tuning
sft: experiment/sft/$(name)/.done_sft

# rlhf training (grpo / dapo) — the RECAP training entry point
grpo: experiment/grpo/$(name)/.done_grpo

# batch safety evaluation (StrongReject-Prefill + WildJailbreak-harm)
batch_eval:
> @echo "[INFO] Running batch evaluation (StrongReject, StrongReject Prefill) for model: $(name)"
>	$(MAKE) oai_safety_judge name=$(name)_strongrejdsqwen1b
> $(MAKE) oai_safety_judge name=$(name)_wjbharm

# used for monitoring variables
print-%:
> @echo '$* = $($*)'


help:
> @echo "Usage: make [target] name=<experiment>"
> @echo "Targets: venv, sft, grpo, infer, guard, batch_eval"
> @echo "Example: make grpo  name=dsllama_8b_mathgsmharmchainprestarorpre"
> @echo "         make infer name=dsllama_8b_strongrej"
> @echo "         make guard name=dsllama_8b_strongrej guardrail=granite_guardian31_8b"
> @echo "         make sft   name=dsqwen_7b_harmchain1ksft"
> @echo "         make batch_eval name=dsllama_8b"
.PHONY: default venv sft grpo infer guard oai_safety_judge oai_or_judge batch_eval help
