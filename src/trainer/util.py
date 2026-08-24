"""Small stateless helpers used by the RLHF rollout/training loop, including
two DAPO-specific pieces: soft overlong-response reward shaping and dynamic
sampling (dropping zero-signal prompt groups), plus generic Ray-future and
tensor-dict utilities.
"""

from typing import Any, Dict, List

import ray
import torch

from src.data import Experience

__all__ = [
    "compute_overlong_reward",
    "flatten_tensor_dict",
    "resolve_dict_futures",
    "get_dynamic_sampling_experiences",
]


def compute_overlong_reward(
    exceed_lens: List,
    buffer_len: int,
    penalty_factor: float,
) -> List:
    """DAPO Soft Overlong Punishment.

    For each rollout, `exceed_lens` is how many tokens its response ran past
    the "safe" length budget (generate_max_len - overlong_buffer_len); it may
    be negative for responses that finished within budget. Rather than a hard
    cutoff, this linearly scales a penalty over the overlong buffer window and
    clips it to be non-positive, so overlong responses are softly discouraged
    without discarding them.

    Args:
        exceed_lens: per-rollout number of tokens beyond the safe length budget.
        buffer_len: length of the overlong buffer window used to scale the penalty.
        penalty_factor: maximum penalty magnitude applied at/after a full buffer overrun.

    Returns:
        Per-rollout penalty values in (-inf, 0], to be added to the reward.
    """
    overlong_rewards = [
        min(-exceed_len / buffer_len * penalty_factor, 0) for exceed_len in exceed_lens
    ]
    return overlong_rewards


def get_dynamic_sampling_experiences(
    experiences: List[Experience],
    n_samples_per_prompt: int,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
):
    """DAPO dynamic sampling: drop prompt groups with no learning signal.

    `experiences` is assumed to be laid out as consecutive groups of
    `n_samples_per_prompt` rollouts sampled from the same prompt (as GRPO/DAPO
    do). A group's average reward near the extremes (e.g. all rollouts get
    reward 0, or all get reward 1) yields ~zero advantage/gradient for GRPO, so
    such groups are filtered out; only groups whose average reward strictly
    falls within (min_reward, max_reward) are kept.

    Args:
        experiences: rollouts, grouped contiguously by prompt.
        n_samples_per_prompt: number of rollouts sampled per prompt (group size).
        min_reward: exclusive lower bound on a group's average reward to keep it.
        max_reward: exclusive upper bound on a group's average reward to keep it.

    Returns:
        The flattened list of experiences belonging to kept groups.
    """
    filtered_samples = []
    min_reward, max_reward = float(min_reward), float(max_reward)
    # Group individual samples into batches of n_samples size
    for i in range(0, len(experiences), n_samples_per_prompt):
        batch_samples = experiences[i : i + n_samples_per_prompt]
        assert len(batch_samples) == n_samples_per_prompt

        # Calculate average reward for this batch of samples
        avg_reward = (
            sum(sample.rewards.item() for sample in batch_samples)
            / n_samples_per_prompt
        )

        # Check if average reward is within the specified range
        if min_reward < avg_reward < max_reward:
            filtered_samples.extend(batch_samples)
    return filtered_samples


def flatten_tensor_dict(
    data: Dict[str, List[torch.Tensor]], duplicate_factor: int = 1
) -> Dict[str, List[List[torch.Tensor]]]:
    """For each key, drop redundant duplicate entries (every `duplicate_factor`-th
    sublist is kept, e.g. to undo repeated results from ring-attention/tensor-parallel
    workers) and flatten the remaining tensors into a single list of scalar tensors.
    """

    def flatten_tensor_list(
        tensor_list: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        ret = [item for sublist in tensor_list[::duplicate_factor] for item in sublist]
        return [t for tensor in ret for t in tensor.view(-1)]

    return {k: flatten_tensor_list(v) for k, v in data.items()}


def resolve_dict_futures(future_dict: Dict[Any, List[ray.ObjectRef]]) -> Dict[Any, Any]:
    """
    Resolve all Ray futures in a dictionary and return a new dictionary
    with the same keys and resolved values.

    Args:
        future_dict (Dict[Any, ray.ObjectRef]): A dictionary where values are Ray futures.

    Returns:
        Dict[Any, Any]: A dictionary with the same keys and resolved values.
    """
    return {k: ray.get(v) for k, v in future_dict.items()}
