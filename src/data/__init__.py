"""Data loading, prompt formatting, and RL rollout bookkeeping.

Exposes the Hydra-driven dataset (:class:`HydraConfigDataset`, which also
applies counter-aligned CoT *prefills* when ``data.use_prefill`` is set), the
per-rollout :class:`Experience` container, the replay buffer, and
dataset-blending/utility helpers.
"""

from .dataset import HydraConfigDataset
from .experience import Experience
from .replay_buffer import *
from .util import *
