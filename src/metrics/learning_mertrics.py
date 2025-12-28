"""Evaluation helpers for experiments."""

from __future__ import annotations

from typing import Optional, Tuple


def total_reward_per_episode(rewards: list[float]) -> float:
    """Return the total reward for an episode."""
    return float(sum(rewards))
