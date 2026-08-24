"""MMLU multiple-choice accuracy scoring.

Benchmark metric only: extracts a single A/B/C/D answer letter from the
model's generation and compares it against the label.
"""

import re
from typing import Any, Dict, List


__all__ = ["normalize_mmlu_answer", "mmlu_accuracy"]


def normalize_mmlu_answer(text: Any):
    """Normalize a raw MMLU answer/generation to a bare a/b/c/d letter.

    Lowercases, strips whitespace/punctuation, drops common prefixes like
    "the answer is"/"answer:", then (if more than one character remains)
    strips everything but a-d.
    """
    assert isinstance(text, str), f"The input should be string, not {type(text)}"
    # Lowercase for consistency
    text = text.lower()

    # Remove whitespace, newlines, periods, commas
    text = re.sub(r"[\s\.,]", "", text)

    # Remove common LLM artifacts
    # (e.g., "The answer is A", "Answer: A", etc.)
    text = re.sub(r"(theansweris|answer:)", "", text)

    # keep only single character (A/B/C/D)
    # You can enable this if you're confident the answer is always a single letter
    if len(text) > 1:
        text = re.sub(r"[^a-d]", "", text)

    return text


def mmlu_accuracy(
    label_prediction_pairs: List[Dict[str, str]],
    label_name: str = "label",
    prediction_name: str = "generation",
) -> float:
    """
    label_prediction_pairs - the content dumped to the jsonl file
    label_prediction_pairs: [{"prompt": xxx, "generation": xxx}]

    Returns (accuracy_pct, num_correct, num_total).
    """

    num_correct = 0
    num_total = len(label_prediction_pairs)
    for sample in label_prediction_pairs:
        assert isinstance(sample, dict)
        assert label_name in sample and prediction_name in sample

        label = normalize_mmlu_answer(sample[label_name])
        assert label in "abcd"

        prediction = normalize_mmlu_answer(sample[prediction_name])

        num_correct += label == prediction

    return num_correct / num_total * 100.0, num_correct, num_total
