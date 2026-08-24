"""DeepSpeed helper for wrapping a model in an ``init_inference`` engine.

Used to accelerate generation/inference. Note this is distinct from the ZeRO
*training* strategy, which is configured via OpenRLHF's ``DeepspeedStrategy``.
"""

from .strategy import *