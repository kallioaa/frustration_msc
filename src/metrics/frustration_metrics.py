"""Frustration metrics for TD learning."""

from __future__ import annotations

import math

from functools import partial
from typing import Any


def frustration_metrics_specs() -> dict[str, Any]:
    frustration_metrics = {
        "total_td_error_per_episode": total_td_error_per_episode,
        "mean_negative_td_error_per_episode": mean_negative_td_error_per_episode,
        "mean_absolute_td_error_per_episode": mean_absolute_td_error_per_episode,
        "negative_td_error_sum_per_episode": negative_td_error_sum_per_episode,
        "frustration_rate_per_episode": frustration_rate_per_episode,
        "tail_frustration_per_episode": partial(
            tail_frustration_per_episode, percentile=0.90
        ),
        "cvar_tail_frustration_per_episode": partial(
            cvar_tail_frustration_per_episode, percentile=0.90
        ),
    }
    return frustration_metrics


def total_td_error_per_episode(td_errors: list[float]) -> float:
    """Return the total TD error for an episode."""
    return float(sum(td_errors))


def mean_absolute_td_error_per_episode(td_errors: list[float]) -> float:
    """Return the mean absolute TD error for an episode."""
    if not td_errors:
        return 0.0
    return float(sum(abs(float(td_error)) for td_error in td_errors) / len(td_errors))


def frustration_rate_per_episode(td_errors: list[float]) -> float:
    """Return the percentage of negative td_errors"""
    if not td_errors:
        raise ValueError("td_errors must be non-empty")

    n = len(td_errors)
    n_negative = sum(1 for td_error in td_errors if td_error < 0)

    result = n_negative / n if n > 0 else 0.0
    return result


def mean_negative_td_error_per_episode(td_errors: list[float]) -> float:
    """Return the mean of negative TD errors for an episode."""
    negative_errors = [float(td_error) for td_error in td_errors if td_error < 0]
    if not negative_errors:
        return 0.0

    return sum(negative_errors) / len(negative_errors)


def negative_td_error_sum_per_episode(td_errors: list[float]) -> float:
    """Return the accumulated negative TD error over an episode."""
    if not td_errors:
        return 0.0
    return float(sum(td_error for td_error in td_errors if td_error < 0))


def tail_frustration_per_episode(
    td_errors: list[float], percentile: float = 0.9
) -> float:
    """Compute tail frustration from max(0, -td_error) within an episode.

    Args:
        td_errors: Per-step TD errors for a single episode.
        percentile: Percentile in [0, 1] (e.g., 0.9 for 90th).
    """
    if not td_errors:
        raise ValueError("td_errors must be non-empty")
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be between 0 and 1")

    magnitudes = [min(0.0, float(td_error)) for td_error in td_errors]
    if all(value == 0.0 for value in magnitudes):
        return 0.0

    magnitudes.sort()
    n = len(magnitudes)

    index = (n - 1) * percentile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return magnitudes[lower]
    weight = index - lower
    return magnitudes[lower] * (1.0 - weight) + magnitudes[upper] * weight


def cvar_tail_frustration_per_episode(
    td_errors: list[float], percentile: float = 0.9
) -> float:
    """Compute CVaR-style mean of the worst tail of max(0, -td_error).

    Args:
        td_errors: Per-step TD errors for a single episode.
        percentile: Percentile in [0, 1] (e.g., 0.9 for 90th).
    """
    if not td_errors:
        raise ValueError("td_errors must be non-empty")
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be between 0 and 1")

    magnitudes = [min(0.0, float(td_error)) for td_error in td_errors]
    if all(value == 0.0 for value in magnitudes):
        return 0.0

    magnitudes.sort()
    n = len(magnitudes)
    tail_fraction = max(1, math.ceil((1.0 - percentile) * n))
    tail_values = magnitudes[-tail_fraction:]
    return sum(tail_values) / len(tail_values)
