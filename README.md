# RECAP: Large Reasoning Models Learn Better Alignment from Flawed Thinking [EMNLP'26 Main]

[![arXiv](https://img.shields.io/badge/arXiv-2510.00938-red)](https://arxiv.org/abs/2510.00938)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP'26-Main-blueviolet)](https://arxiv.org/abs/2510.00938)
[![License: MIT](https://img.shields.io/badge/License-MIT-success)](./LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

<p align="center">
  <img src="figures/method.png" width="100%" alt="RECAP training pipeline: prompts are prefilled with counter-aligned CoT so the policy learns to override flawed trajectories.">
</p>

**RECAP (Robust Safety Alignment via Counter-Aligned Prefilling)** is an RL post-training method that teaches large reasoning models (LRMs) to override *flawed* chain-of-thought (CoT) and reroute to safe, helpful answers. During RL rollouts, a fraction of prompts are **prefilled with counter-aligned reasoning** — unsafe CoT for harmful prompts and over-conservative refusal CoT for benign ones. Because the injected prefix is treated as fixed context, the policy is optimized only on the tokens it generates *after* it, so to earn reward it must recognize and correct the flawed opening rather than follow it. RECAP drops into a standard RLHF (GRPO/DAPO) pipeline with **no extra optimization steps, architectural changes, or inference-time cost**, and substantially improves safety and jailbreak robustness while *reducing* overrefusal and preserving reasoning ability.

This is the official code repository for [Large Reasoning Models Learn Better Alignment from Flawed Thinking](https://arxiv.org/abs/2510.00938) (EMNLP 2026, Main).

## How RECAP works

As shown above, each RL batch mixes three kinds of prompts: **benign prompts prefilled with refusal CoT** (sampled from an overly-conservative model), **harmful prompts prefilled with unsafe CoT** (sampled from a weakly-aligned model), and **standard prompts with no prefill**. The reward is computed on the final response, and only the model-generated continuation is optimized — so the gradient pushes the policy to reroute away from the flawed opening (e.g. *"But wait, it seems illegal and unethical…"* on a harmful prompt, or *"Instead, I'll offer alternative solutions…"* on a benign one).

The result is a model that reasons critically about safety instead of inheriting the stance of its opening tokens: it **stops unsafe reasoning on harmful prompts and corrects overly conservative reasoning on benign ones.**

<p align="center">
  <img src="figures/example.png" width="100%" alt="Vanilla RLHF vs. RECAP on a harmful prompt and a benign prompt.">
</p>

On a **harmful** prompt (top), vanilla RLHF decodes the request and continues toward compliance, while RECAP catches the intent mid-reasoning and refuses with a constructive redirect. On a **benign** prompt (bottom), vanilla RLHF overrefuses (*"I'm sorry, but I can't assist with that"*), while RECAP recovers and answers helpfully.

## Quick Start

### Installation

Create the conda environment and install everything (Python 3.10):

```bash
make venv
conda activate llm
```

Or install into an existing Python 3.10 environment:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install <flash-attn wheel matching your CUDA/torch/python>   # see FLASH_ATTN_WHEEL in the Makefile
pip install -r requirements.txt
pip install -e .
```

### Training

**Supervised Fine-Tuning (SFT)**
```bash
make sft name=<model>_<dataset>
```

**RLHF / GRPO (RECAP training)**
```bash
make grpo name=<model>_<dataset>
```

### Inference

**Batch inference**
```bash
make infer name=<model>_<dataset>
```

### Evaluation

**Safety evaluation with guardrail models**
```bash
make guard name=<model>_<dataset> guardrail=<guard_model>
```

### Generating counter-aligned prefills

RECAP trains on a mixture of standard prompts and prompts whose chain-of-thought
(CoT) is prefilled with **counter-aligned** reasoning (unsafe CoT for harmful
prompts, over-conservative refusal CoT for benign ones). Building these prefill
datasets is a two-stage process.

**Stage 1 — sample the CoT and filter it with a guardrail.** Use a weakly-aligned
generator model to produce CoT, then score it. This writes
`experiment/infer/<name>/generation_<name>.jsonl` and `guardrail_<name>.jsonl`:
```bash
make infer name=<generator>_<dataset>                       # sample CoT
make guard name=<generator>_<dataset> guardrail=<guard>     # score / filter it
```

**Stage 2 — slice the CoT into prefill columns** with `src/generate_prefills.py`.
It reads the generated CoT and writes the `*_prefill.jsonl` files that the
training targets consume. **No paths are hardcoded** — point every input/output
with a flag (change the `--harm1k` / `--generation` / `--judge` / `--dataset` /
`--output` values to your own locations):

```bash
# harmful prompts -> pre100/300/500/700 + r025/050/075/100 columns
python -m src.generate_prefills harmchain \
    --harm1k data/harmchain/harm1k.jsonl \
    --output data/harmchain/rlhf_prefill.jsonl

# STAR-1 safety prompts -> pre200  (--generation is your Stage-1 output)
python -m src.generate_prefills star1k \
    --generation experiment/infer/<name>/generation_<name>.jsonl \
    --output data/star1k/prefill_unsafe.jsonl

# STAR-benign overrefusal prompts -> pre300
python -m src.generate_prefills star_or \
    --generation experiment/infer/<name>/generation_<name>.jsonl \
    --output data/star_or/prefill_rej.jsonl

# adversarial-benign prompts -> pre_500_r50
python -m src.generate_prefills wildjailbreak \
    --generation experiment/infer/<name>/generation_<name>.jsonl \
    --judge experiment/infer/<name>/guardrail_<name>.jsonl \
    --dataset data/wildjailbreak/adversarial_benign.jsonl \
    --output data/wildjailbreak/adversarial_benign_prefill.jsonl
```

Prefill length, prefill ratio, and (for `harmchain`) the per-column word budgets
are all flags with paper defaults — run `python -m src.generate_prefills <cmd> -h`
to see them. The resulting files are picked up by the `harmchain*` / `staror*`
training datasets (see `makefiles/dataset.mk`).

## Supported Models

| Family | Sizes |
|---|---|
| LLaMA (2 / 3 / 3.1 / 3.2 / 3.3) | 1B – 70B |
| Mistral | 24B |
| Qwen | 1.5B – 32B |
| Gemma | 2B – 27B |
| DeepSeek-R1 distills | 1.5B – 32B |
| Granite, Phi, STAR-1, SafeChain | various |

## Datasets & Benchmarks

**Training** (harmful / overrefusal / math): BeaverTails, HarmChain, STAR-1 / STAR-benign, GSM8K, MATH. Training prompts and counter-aligned prefills are **not bundled** in this repo — pull the source prompts from HuggingFace and build the prefills with `src/generate_prefills.py` (see "Generating counter-aligned prefills" above).

**Safety evaluation** (bundled under `data/`): StrongReject (+ prefilled variant), AdvBench, WildJailbreak, XSTest, Fortress.

**Knowledge & reasoning**: GSM8K, MATH500, AIME.

**Guardrail / judge models**: LlamaGuard, Granite Guardian, WildGuard, GPT-4o judge.

## Configuration

All configs are managed via [Hydra](https://hydra.cc/) under the `config/` directory, covering dataset, model, training, logging, and inference settings. Experiments are launched through `make` targets (see `makefiles/`); run artifacts and checkpoints are written under `experiment/`.

## Citation

```bibtex
@article{peng2025recap,
  title={Large Reasoning Models Learn Better Alignment from Flawed Thinking},
  author={Peng, ShengYun and Smith, Eric and Evtimov, Ivan and Jiang, Song and Chen, Pin-Yu and Zhan, Hongyuan and Wang, Haozhu and Chau, Duen Horng and Pasupuleti, Mahesh and Chi, Jianfeng},
  journal={arXiv preprint arXiv:2510.00938},
  year={2025}
}
```
