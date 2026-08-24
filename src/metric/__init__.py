"""Evaluation metrics and reward-scoring functions.

Groups the scorers used both as RL reward signals and for benchmark evaluation:
guardrail-based safety scoring, the GPT-4o / rubric overrefusal (helpfulness)
judges, WildGuard, and verifiable math / multiple-choice accuracy (GSM8K,
MATH500, AIME, MMLU, ARC). :func:`compute_metric` dispatches to the right scorer
for a given benchmark id.
"""

import logging
from typing import Dict, List

from .arc_challenge import *
from .guardrail import *
from .keyword import *
from .math import *
from .mmlu import *
from .oai_judge import *
from .util import *
from .wildguard import *
from .overrefusal import *

logger = logging.getLogger(__name__)


def compute_metric(
    dataset_id: str,
    label_prediction_pairs: List[Dict[str, str]],
    label_name: str = "label",
    prediction_name: str = "generation",
    model_name: str = "default",
):
    dataset_id = dataset_id.lower().strip()

    if dataset_id in ["mmlu"]:
        return mmlu_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    elif dataset_id in ["gsm8k", "gsm8k_platinum", "aime24", "math500"]:
        return math_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    elif dataset_id in ["arc_challenge"]:
        return arc_challenge_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    elif dataset_id in ["star_or", "xstest", "fortress_or", "wildjailbreak"]:
        return wildguard_generation_score(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    else:
        logger.info("Using guardrail safety evaluation")
        return guardrail_safety_score(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
            model_name=model_name,
        )
