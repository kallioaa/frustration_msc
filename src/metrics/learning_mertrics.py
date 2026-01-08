"""Evaluation helpers for experiments."""

from __future__ import annotations

from typing import Optional, Tuple


def total_reward_per_episode(rewards: list[float]) -> float:
    """Return the total reward for an episode."""
    return float(sum(rewards))


def episode_won(rewards: list[float], win_threshold: float = 1.0) -> float:
    """Return 1.0 if the last reward indicates goal reached, else 0.0."""
    if not rewards:
        return 0.0
    return 1.0 if rewards[-1] >= win_threshold else 0.0
