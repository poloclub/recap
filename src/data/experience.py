"""The :class:`Experience` container: a batch of rollout data (tokens, masks,
log-probs, rewards, advantages, KL, ...) produced by rollout/experience-making
and consumed by RLHF policy/critic training.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union

import torch
# `to` recursively moves tensors (including nested lists of tensors) to a device
from openrlhf.trainer.ppo_utils.experience_maker import to

from .util import zero_pad_sequences

__all__ = ["Experience"]


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    conversation: List[List[Dict]]
    """

    # unique id
    rollout_ids: int = None

    # saved by dataset
    conversations: List[List[Dict[str, str]]] = None  # assistant is the label
    reward_names: List[str] = None  # determines which RM to call

    # saved by sampling generator
    sequences: torch.Tensor = None  # tokenized input+output
    generation_texts: List[str] = None
    exceed_lens: int = None  # used for  Soft Overlong Punishment in DAPO

    # saved by experience maker
    attention_masks: torch.LongTensor = None
    action_masks: torch.BoolTensor = None

    # policy + ref
    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None

    # saved by RM
    rewards: torch.Tensor = None
    advantages: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None

    # kl
    kl: torch.Tensor = None

    # store additional information, all the fields will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            # the "info" field is a dict of tensors/values, move each value individually
            if isinstance(value, dict):
                setattr(
                    self, field, {key: to(val, device) for key, val in value.items()}
                )
            else:
                setattr(self, field, to(value, device))

        return self

    @staticmethod
    def _merge_item(
        items: List, pad_value: int = 0
    ) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {
                key: Experience._merge_item(values, pad_value)
                for key, values in result.items()
            }
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(
        experiences_list: List["Experience"], pad_token_id
    ) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)

    @staticmethod
    def split_experience_batch(experience: "Experience") -> List["Experience"]:
        """Split a batch experience into a list of individual experiences. drop padding tokens if exists"""
        if not experience:
            return []

        batch_size = len(experience.sequences)
        keys = [f.name for f in fields(Experience) if f.name != "info"]

        # Sanity check on field lengths
        for key in keys:
            value = getattr(experience, key)
            if value is not None and len(value) != batch_size:
                raise ValueError(
                    f"Size of {key} ({len(value)}) does not match batch_size ({batch_size})"
                )

        # Sanity check on info
        for k, v in experience.info.items():
            if not isinstance(v, (torch.Tensor, list)) or len(v) != batch_size:
                raise ValueError(
                    f"Invalid or mismatched size for info[{k}]: {type(v)}, len={len(v)}"
                )

        # Split into List[Experience]
        experiences = [
            Experience(
                **{
                    key: getattr(experience, key)[i]
                    if getattr(experience, key) is not None
                    else None
                    for key in keys
                },
                info={k: v[i] for k, v in experience.info.items()},
            )
            for i in range(batch_size)
        ]

        experiences = [
            Experience.remove_padding_in_experience(exp) for exp in experiences
        ]

        return experiences

    @staticmethod
    def remove_padding_in_experience(experience: "Experience") -> "Experience":
        """Trim trailing right-padding off a single (unbatched) Experience,
        using its attention_masks to find how many pad tokens to drop."""
        keys_to_trim = [
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_masks",
            "action_masks",
        ]

        # Find first non-padding token from the end
        right_pad = experience.attention_masks.flip(0).argmax().item()
        end = None if right_pad == 0 else -right_pad

        # Trim each relevant tensor if it's not None
        for key in keys_to_trim:
            tensor = getattr(experience, key)
            if tensor is not None:
                setattr(experience, key, tensor[:end])

        return experience
