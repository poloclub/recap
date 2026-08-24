"""Top-level package for the LLM post-training and evaluation toolkit.

This repository provides a unified, Hydra-configured pipeline for:

* **Training** — supervised fine-tuning (:mod:`src.sft`), direct preference
  optimization (:mod:`src.dpo_openrlhf`), and online RLHF with GRPO/DAPO
  (:mod:`src.rlhf`). The RLHF path is what powers **RECAP** (Robust Safety
  Alignment via Counter-Aligned Prefilling): counter-aligned chain-of-thought
  (CoT) is injected as a fixed prefix into a fraction of the rollouts and the
  policy is optimized only over the tokens it generates *after* that prefix,
  teaching it to override flawed reasoning.
* **Inference** — batched generation with vLLM (:mod:`src.infer_vllm`) or
  HuggingFace (:mod:`src.infer_hf`), plus interactive chat helpers.
* **Evaluation** — safety/helpfulness/reasoning scoring via guardrail reward
  models and an OpenAI (GPT-4o) judge (:mod:`src.oai_judge`, :mod:`src.metric`).

Experiments are orchestrated through ``make`` targets that assemble Hydra
overrides from ``config/`` and ``makefiles/``.
"""
