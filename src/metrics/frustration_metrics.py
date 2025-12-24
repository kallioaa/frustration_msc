"""Frustration metrics for TD learning."""

from __future__ import annotations

from typing import Iterable


def negative_td_errors(td_errors: Iterable[float]) -> list[float]:
    """Return a list with negative TD errors, zeros otherwise."""
    result: list[float] = []
    for td_error in td_errors:
        value = float(td_error)
        result.append(value if value < 0.0 else 0.0)
    return result
