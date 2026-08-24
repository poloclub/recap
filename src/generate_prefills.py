"""Generate RECAP counter-aligned CoT prefill datasets (stage 2 of prefill prep).

RECAP injects a fixed, counter-aligned chain-of-thought (CoT) prefix into a
fraction of the RL rollouts so the policy learns to override flawed reasoning.
Building those prefills is a two-stage process:

  Stage 1 (sample + filter the CoT)   -- done via the existing pipeline:
      make infer  name=<generator>_<dataset>      # sample CoT from a generator model
      make guard  name=<generator>_<dataset> ...  # score it with a guardrail
    This writes experiment/infer/<name>/generation_<name>.jsonl and
    guardrail_<name>.jsonl.

  Stage 2 (slice CoT into prefill columns)  -- THIS script:
    Reads the generated CoT, truncates it to fixed word lengths, assigns the
    per-prompt prefill ratio, and writes the *_prefill.jsonl files consumed by
    the training make targets (harmchain / staror / etc.).

No paths are hardcoded: every input and output is a CLI flag, so this runs
anywhere. Defaults reproduce the datasets used in the paper. See the README
section "Generating counter-aligned prefills" for the end-to-end recipe.

Subcommands (one per training dataset):
    harmchain      harmful prompts + unsafe CoT   -> pre100/300/500/700, r025/050/075/100
    star1k         STAR-1 safety                  -> pre200
    star_or        STAR-benign overrefusal        -> pre300
    wildjailbreak  adversarial-benign             -> pre_500_r50

Example:
    python -m src.generate_prefills harmchain \
        --harm1k data/harmchain/harm1k.jsonl \
        --output data/harmchain/rlhf_prefill.jsonl
"""

import argparse
from typing import Dict, List

import jsonlines
from datasets import load_dataset

# DeepSeek-distilled reasoning models open their CoT with this token; every
# prefill string starts here so the model continues *inside* the think block.
THINK_OPEN = "<think>\n"


def _prefix(cot: str, n_words: int) -> str:
    """Return an opened think-block seeded with the first ``n_words`` of ``cot``."""
    return f"{THINK_OPEN}{' '.join(cot.split(' ')[:n_words])}"


def _empty_prefix() -> str:
    """Return an empty (un-prefilled) think block: the model reasons from scratch."""
    return THINK_OPEN


def _read_jsonl(path: str) -> List[Dict]:
    with jsonlines.open(path, "r") as f:
        return list(f)


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    with jsonlines.open(path, "w") as f:
        f.write_all(rows)


def build_harmchain(args: argparse.Namespace) -> None:
    """Harmful prompts -> unsafe-CoT prefills with length and ratio variants.

    Length columns ``pre{L}`` prefill the first ``args.pre_fraction`` of prompts
    with the first ``L`` words of the (already sampled + guard-filtered) unsafe
    CoT. Ratio columns ``r{XX}`` instead fix the prefill length at
    ``args.ratio_length`` words and vary the *fraction* of prompts that get a
    prefill (XX%), leaving the rest to reason from scratch.
    """
    data = _read_jsonl(args.harm1k)  # each row must carry a "cot" field
    n = len(data)

    # length variants: prefill the same leading fraction of prompts, vary length
    pre_cut = round(args.pre_fraction * n)
    for idx, d in enumerate(data):
        for length in args.pre_lengths:
            d[f"pre{length}"] = (
                _prefix(d["cot"], length) if idx < pre_cut else _empty_prefix()
            )

    # ratio variants: fix the length, vary the fraction of prompts prefilled
    for ratio in args.ratios:
        col = f"r{round(ratio * 100):03d}"
        cut = round(ratio * n)
        for idx, d in enumerate(data):
            d[col] = (
                _prefix(d["cot"], args.ratio_length)
                if idx < cut
                else _empty_prefix()
            )

    _write_jsonl(args.output, data)
    print(f"[harmchain] wrote {n} rows -> {args.output}")


def build_from_hf(args: argparse.Namespace) -> None:
    """STAR-1 / STAR-benign prompts -> single fixed-length prefill column.

    Loads prompts from a HuggingFace dataset and aligns them by index with a
    generation file (the sampled CoT). The first ``args.pre_fraction`` of prompts
    get a ``pre{args.pre_length}`` prefill built from that CoT; the rest get an
    empty think block. Used for both the STAR-1 unsafe prefills and the
    STAR-benign refusal prefills (they differ only in defaults).
    """
    prompts = load_dataset(args.hf_dataset, split=args.split)
    gen = _read_jsonl(args.generation)
    cut = round(args.pre_fraction * len(prompts))
    col = f"pre{args.pre_length}"

    def add_prefill(sample: Dict, idx: int) -> Dict:
        g = gen[idx] if idx < len(gen) else {}
        # generation files store CoT under "cot"; some fall back to "generation"
        cot = g.get("cot", g.get("generation") if args.allow_generation_fallback else None)
        if idx < cut and cot is not None:
            sample[col] = _prefix(cot, args.pre_length)
        else:
            sample[col] = _empty_prefix()
        return sample

    prompts = prompts.map(add_prefill, with_indices=True)
    prompts.to_json(args.output, lines=True)
    print(f"[{args.hf_dataset}] wrote {len(prompts)} rows -> {args.output}")


def build_wildjailbreak(args: argparse.Namespace) -> None:
    """Adversarial-benign prompts -> refusal-CoT prefill (over-conservative).

    Keeps only prompts the guardrail judge marked as refused (``refusal_marker``)
    and whose generation carries a CoT, attaches that CoT, then prefills the
    first ``args.pre_fraction`` of the survivors with the first ``pre_length``
    words. This produces the over-conservative reasoning RECAP trains the model
    to override on benign inputs.
    """
    generations = _read_jsonl(args.generation)
    judges = _read_jsonl(args.judge)
    dataset = _read_jsonl(args.dataset)

    kept: List[Dict] = []
    for g, j, d in zip(generations, judges, dataset):
        if args.refusal_marker in j["generation"] and "cot" in g:
            d = dict(d)
            d["cot"] = g["cot"]
            kept.append(d)

    cut = round(args.pre_fraction * len(kept))
    for idx, d in enumerate(kept):
        d[args.column] = (
            _prefix(d["cot"], args.pre_length) if idx <= cut else _empty_prefix()
        )

    _write_jsonl(args.output, kept)
    print(
        f"[wildjailbreak] kept {len(kept)}/{len(dataset)} refused rows -> {args.output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RECAP counter-aligned CoT prefill datasets (stage 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- harmchain: harmful prompts + unsafe CoT (length + ratio variants) ----
    p = sub.add_parser("harmchain", help="harmful unsafe-CoT prefills")
    p.add_argument("--harm1k", required=True, help="input jsonl with a 'cot' field")
    p.add_argument("--output", required=True, help="output *_prefill.jsonl path")
    p.add_argument("--pre-lengths", type=int, nargs="+", default=[100, 300, 500, 700],
                   help="word lengths for the pre{L} columns")
    p.add_argument("--pre-fraction", type=float, default=0.5,
                   help="fraction of prompts prefilled for the pre{L} columns")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00],
                   help="prompt fractions for the r{XX} columns")
    p.add_argument("--ratio-length", type=int, default=500,
                   help="fixed word length used by the r{XX} columns")
    p.set_defaults(func=build_harmchain)

    # ---- star1k: STAR-1 safety prompts + generated CoT ----
    p = sub.add_parser("star1k", help="STAR-1 unsafe prefills")
    p.add_argument("--hf-dataset", default="UCSC-VLAA/STAR-1", help="HF prompt dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--generation", required=True,
                   help="generation jsonl aligned by index (holds 'cot'/'generation')")
    p.add_argument("--output", required=True, help="output prefill_unsafe.jsonl path")
    p.add_argument("--pre-length", type=int, default=200)
    p.add_argument("--pre-fraction", type=float, default=0.5)
    p.add_argument("--allow-generation-fallback", action="store_true", default=True,
                   help="fall back to the 'generation' field when 'cot' is absent")
    p.set_defaults(func=build_from_hf)

    # ---- star_or: STAR-benign overrefusal prompts + refusal CoT ----
    p = sub.add_parser("star_or", help="STAR-benign refusal prefills")
    p.add_argument("--hf-dataset", default="UCSC-VLAA/STAR-benign-915", help="HF prompt dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--generation", required=True,
                   help="generation jsonl aligned by index (holds 'cot')")
    p.add_argument("--output", required=True, help="output prefill_rej.jsonl path")
    p.add_argument("--pre-length", type=int, default=300)
    p.add_argument("--pre-fraction", type=float, default=0.5)
    p.add_argument("--allow-generation-fallback", action="store_true", default=False,
                   help="only prefill rows whose generation carries a 'cot' field")
    p.set_defaults(func=build_from_hf)

    # ---- wildjailbreak: adversarial-benign prompts + refusal CoT ----
    p = sub.add_parser("wildjailbreak", help="adversarial-benign refusal prefills")
    p.add_argument("--generation", required=True, help="generation jsonl (holds 'cot')")
    p.add_argument("--judge", required=True, help="guardrail jsonl aligned by index")
    p.add_argument("--dataset", required=True, help="adversarial_benign.jsonl prompts")
    p.add_argument("--output", required=True,
                   help="output adversarial_benign_prefill.jsonl path")
    p.add_argument("--column", default="pre_500_r50", help="name of the prefill column")
    p.add_argument("--pre-length", type=int, default=500)
    p.add_argument("--pre-fraction", type=float, default=0.5)
    p.add_argument("--refusal-marker", default="Response refusal: yes",
                   help="substring in the judge output marking a refusal")
    p.set_defaults(func=build_wildjailbreak)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
