"""Shared low-level helpers for metric/reward modules."""

import logging
from typing import List, Union

from vllm.sequence import Logprob

logger = logging.getLogger(__name__)

__all__ = ["get_vllm_logprob_or_last"]


def get_vllm_logprob_or_last(
    logprobs_list: List[Logprob],
    target_tokens: Union[str, List[str]],
) -> float:
    """return the logprobs of all the target_tokens, default to the logprobs of the last token"""
    # logprobs_list: vllm Logprob objects for one decoding step (the top-k
    # candidates and their logprobs).
    # target_tokens: token string(s) to look up.
    # Returns a list of logprobs, one per target token; if a target token is
    # not among the top-k candidates, falls back to the logprob of the last
    # listed candidate (an approximation, not necessarily that token's true
    # logprob).
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens]

    # Build a dict: token -> logprob
    token_to_logprob = {item.decoded_token: item.logprob for item in logprobs_list}
    fallback_logprob = logprobs_list[-1].logprob if logprobs_list else float("-inf")

    # Extract in order
    return [token_to_logprob.get(token, fallback_logprob) for token in target_tokens]
