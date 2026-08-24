"""Replay buffer that accumulates rollout :class:`Experience` batches for
policy/critic training, plus the helper that re-pads and stacks individual
Experiences back into a single training batch.
"""

from abc import ABC
from dataclasses import fields
from typing import List

import torch

from .experience import Experience
from .util import zero_pad_sequences

__all__ = [
    "NaiveReplayBuffer",
    "make_experience_batch",
]


def make_experience_batch(items: List[Experience], packing_samples=False) -> Experience:
    """Combine individual Experiences into a batch of experiences."""
    if not items:
        raise ValueError("Empty items list")

    first_item = items[0]
    # fields excluded from the zero-pad-and-stack step below; "info" is merged
    # separately further down, the rest are simply not carried over into the batch
    keys_no_padding = [
        "rollout_ids",
        "conversations",
        "reward_names",
        "generation_texts",
        "rewards",
        "info",
        "exceed_lens",
    ]
    keys = [f.name for f in fields(Experience) if f.name not in keys_no_padding]

    # Pad and stack main attributes
    kwargs = {
        key: (
            zero_pad_sequences(
                [getattr(item, key) for item in items], "right", stack=True
            )
            if getattr(first_item, key) is not None
            else None
        )
        for key in keys
    }

    # Process info fields
    info = {}
    for k in first_item.info.keys():
        values = [item.info[k] for item in items]
        types = {type(v) for v in values}
        if len(types) > 1:
            raise TypeError(f"Inconsistent types in info[{k}]")
        info[k] = (
            torch.tensor(values)
            if all(isinstance(v, (int, float)) for v in values)
            else values
        )

    kwargs["info"] = info
    return Experience(**kwargs)


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self,
        limit: int = 0,  # maximum # of experiences that can be stored in the replay buffer, limit <= 0 means unlimited number of experiences used for training
        packing_samples: bool = False,
        cpu_offload: bool = True,  # move stuff to cpu, but looks like most of the experiences are already on cpu
    ) -> None:
        super().__init__()
        self.limit = limit
        self.packing_samples = packing_samples
        self.items: List[Experience] = []

        self.cpu_offload = cpu_offload

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        """Store one Experience (optionally offloading it to CPU first), evicting
        the oldest entries once the buffer exceeds `limit`."""
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        self.items.append(experience)
        if self.limit > 0 and len(self.items) > self.limit:
            overflow = len(self.items) - self.limit
            self.items = self.items[overflow:]

    def clear(self) -> None:
        """Remove all stored experiences."""
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]

    def collate_fn(self, batch: List[Experience]) -> Experience:
        """DataLoader collate function: merge a list of Experiences into one batched Experience."""
        experience = make_experience_batch(batch, self.packing_samples)
        return experience
