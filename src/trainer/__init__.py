"""High-level training loops and rollout generation.

Provides the SFT and online RLHF (:class:`RLHFTrainer`, GRPO/DAPO) driver loops,
plus the rollout machinery (:class:`SamplesGenerator`, :class:`ExperienceMaker`)
that samples completions from vLLM and turns them into reward-scored
:class:`~src.data.Experience` batches.
"""

from .sft_trainer import SFTTrainer
from .rlhf_trainer import RLHFTrainer
from .rollout import *
from .util import *
