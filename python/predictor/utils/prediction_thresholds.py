"""Helpers for mapping prediction probabilities to discrete labels.

Shared between the live prediction service and the 4s-early variant so they
stay aligned with backtester behaviour.
"""

from __future__ import annotations

from typing import Any, Mapping

DEFAULT_SIGNAL_THRESHOLD = 0.50


def resolve_probability_threshold(
    params: Mapping[str, Any] | None,
    default: float = DEFAULT_SIGNAL_THRESHOLD,
) -> float:
    """Return a sanitized decision threshold from strategy params.

    Falls back to ``default`` when the strategy does not define a ``threshold``
    param or when the provided value is invalid/out of range.
    """
    threshold = default
    if params:
        raw = params.get("threshold")
        if raw is not None:
            try:
                threshold = float(raw)
            except (TypeError, ValueError):
                threshold = default
    if not (0.0 < threshold < 1.0):
        threshold = default
    return max(0.01, min(0.99, threshold))


def classify_probability(prob: float, threshold: float) -> int:
    """Convert a probability into {-1, 0, 1} using symmetric thresholds."""
    if prob > threshold:
        return 1
    down_threshold = 1.0 - threshold
    if prob < down_threshold:
        return 0
    return -1


def label_from_prediction(pred: int) -> str:
    if pred == 1:
        return "UP"
    if pred == 0:
        return "DOWN"
    return "UNDEFINED"
