"""Evaluation helpers for experiments."""

from __future__ import annotations

from typing import Optional, Tuple


def total_reward_per_episode(rewards: list[float]) -> float:
    """Return the total reward for an episode."""
    return float(sum(rewards))


def episode_length_per_episode(rewards: list[float]) -> float:
    """Return the number of steps in an episode."""
    return float(len(rewards))


def episode_won_frozenlake(rewards: list[float], win_threshold: float = 1.0) -> float:
    """Return 1.0 if the last reward indicates goal reached, else 0.0."""
    if not rewards:
        return 0.0
    return 1.0 if rewards[-1] >= win_threshold else 0.0


def episode_won_cliffwalking(
    rewards: list[float], cliff_penalty: float = -100.0
) -> float:
    """Return 1.0 if the episode has no cliff falls, else 0.0."""
    return 1.0 if cliff_falls_per_episode(rewards, cliff_penalty) == 0.0 else 0.0


def cliff_falls_per_episode(
    rewards: list[float], cliff_penalty: float = -100.0
) -> float:
    """Return the number of cliff falls in the episode."""
    if not rewards:
        return 0.0
    return float(sum(1 for reward in rewards if reward <= cliff_penalty))


def cliff_fall_rate_per_episode(
    rewards: list[float], cliff_penalty: float = -100.0
) -> float:
    """Return the fraction of steps in the episode that fall into the cliff."""
    if not rewards:
        return 0.0
    falls = cliff_falls_per_episode(rewards, cliff_penalty)
    return float(falls / len(rewards))


def episode_won_taxi_v3(rewards: list[float], win_threshold: float = 20.0) -> float:
    """Return 1.0 if the last reward indicates a successful dropoff, else 0.0."""
    if not rewards:
        return 0.0
    return 1.0 if rewards[-1] >= win_threshold else 0.0


def taxi_illegal_actions_per_episode(
    rewards: list[float], illegal_action_penalty: float = -10.0
) -> float:
    """Return the number of illegal pickup/drop-off actions in Taxi-v3."""
    if not rewards:
        return 0.0
    return float(
        sum(1 for reward in rewards if float(reward) == float(illegal_action_penalty))
    )
