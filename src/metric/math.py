"""Verifiable math answer scoring (GSM8K, MATH500, AIME, etc.).

Uses math_verify / latex2sympy2 to symbolically parse and compare predicted
vs. reference answers (numeric or LaTeX expressions). Used as an RL reward
signal (`math_generation_to_reward`, binary 0/1 per sample) and a benchmark
accuracy metric (`math_accuracy`).
"""

from typing import Dict, List, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

__all__ = ["math_accuracy", "math_generation_to_reward"]


def math_generation_to_reward(
    conversations: List[List[Dict[str, str]]], generations: List[str]
) -> List[float]:
    """Score generated math answers against the reference answer.

    The reference answer is taken from the last message of each conversation
    (assumed to be the ground-truth label). Both reference and generation are
    parsed as LaTeX/expression math and symbolically compared.

    Returns a list of binary rewards (1.0 if the parsed answers verify as
    equal, else 0.0), one per (conversation, generation) pair.
    """
    reward_normalization_config = NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    )

    # Set up extraction config and get regexes
    extraction_target = (
        ExprExtractionConfig(),
        LatexExtractionConfig(normalization_config=reward_normalization_config),
    )
    labels = [
        parse(f"${conversation[-1]['content']}$", extraction_config=extraction_target)
        for conversation in conversations
    ]
    preds = [
        parse(generation, extraction_config=extraction_target)
        for generation in generations
    ]
    rewards = [float(verify(label, pred)) for label, pred in zip(labels, preds)]

    return rewards


def math_accuracy(
    label_prediction_pairs: List[Dict[str, Union[str, List[str]]]],
    label_name: str = "label",
    prediction_name: str = "generation",
) -> float:
    """
    label_prediction_pairs - the content dumped to the jsonl file
    label_prediction_pairs: [{"prompt": xxx, "generation": xxx}]

    Returns (accuracy_pct, num_correct, num_total). If `prediction_name`
    holds a list of generations per sample, the sample counts as correct if
    any of them verifies against the label (pass@k-style scoring).
    """

    acc_normalization_config = NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    )

    # Set up extraction config and get regexes
    extraction_target = (
        ExprExtractionConfig(),
        LatexExtractionConfig(normalization_config=acc_normalization_config),
    )

    num_correct = 0
    num_total = len(label_prediction_pairs)
    for sample in label_prediction_pairs:
        assert isinstance(sample, dict)
        assert label_name in sample and prediction_name in sample

        label = parse(f"${sample[label_name]}$", extraction_config=extraction_target)
        predictions = sample[prediction_name]

        # normalize to a list so both single-generation and multi-generation
        # (pass@k) predictions are handled uniformly
        if isinstance(predictions, str):
            predictions = [predictions]

        matched = any(
            verify(label, parse(pred, extraction_config=extraction_target))
            for pred in predictions
        )
        num_correct += int(matched)

    return num_correct / num_total * 100.0, num_correct, num_total
