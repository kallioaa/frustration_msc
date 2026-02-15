"""Helpers for sweep plot specification presets."""

from __future__ import annotations

from typing import Any

from plots.training_plots import plot_moving_average_multi


def taxi_training_plot_specs() -> list[dict[str, Any]]:
    """Return default training plot specs for Taxi-v3."""
    return [
        {
            "source": "reward",
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "Taxi-v3: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "Taxi-v3: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "episode_won",
            "ylabel": "Success Rate",
            "title": "Taxi-v3: Episode Success",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "taxi_illegal_actions_per_episode",
            "ylabel": "Illegal Actions",
            "title": "Taxi-v3: Illegal Pickup/Drop-off Actions",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "total_td_error_per_episode",
            "ylabel": "TD Error",
            "title": "Taxi-v3: Total TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "mean_absolute_td_error_per_episode",
            "ylabel": "Mean |TD Error|",
            "title": "Taxi-v3: Mean Absolute TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "negative_td_error_sum_per_episode",
            "ylabel": "Negative TD Error Sum",
            "title": "Taxi-v3: Negative TD Error Sum",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "frustration_rate_per_episode",
            "ylabel": "Frustration Rate",
            "title": "Taxi-v3: Frustration Rate",
            "plot_fn": plot_moving_average_multi,
        },
    ]


def cliffwalking_training_plot_specs() -> list[dict[str, Any]]:
    """Return default training plot specs for CliffWalking."""
    return [
        {
            "source": "reward",
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "CliffWalking: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "CliffWalking: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "cliff_falls_per_episode",
            "ylabel": "Cliff Falls",
            "title": "CliffWalking: Falls Per Episode",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "cliff_fall_rate_per_episode",
            "ylabel": "Cliff Fall Rate",
            "title": "CliffWalking: Cliff-Fall Rate",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "total_td_error_per_episode",
            "ylabel": "TD Error",
            "title": "CliffWalking: Total TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "mean_absolute_td_error_per_episode",
            "ylabel": "Mean |TD Error|",
            "title": "CliffWalking: Mean Absolute TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "negative_td_error_sum_per_episode",
            "ylabel": "Negative TD Error Sum",
            "title": "CliffWalking: Negative TD Error Sum",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "frustration_rate_per_episode",
            "ylabel": "Frustration Rate",
            "title": "CliffWalking: Frustration Rate",
            "plot_fn": plot_moving_average_multi,
        },
    ]


def frozenlake_training_plot_specs() -> list[dict[str, Any]]:
    """Return default training plot specs for FrozenLake."""
    return [
        {
            "source": "reward",
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "FrozenLake: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "FrozenLake: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "reward",
            "key": "episode_won",
            "ylabel": "Success Rate",
            "title": "FrozenLake: Episode Success",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "total_td_error_per_episode",
            "ylabel": "TD Error",
            "title": "FrozenLake: Total TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "mean_absolute_td_error_per_episode",
            "ylabel": "Mean |TD Error|",
            "title": "FrozenLake: Mean Absolute TD Error",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "negative_td_error_sum_per_episode",
            "ylabel": "Negative TD Error Sum",
            "title": "FrozenLake: Negative TD Error Sum",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "source": "td_error",
            "key": "frustration_rate_per_episode",
            "ylabel": "Frustration Rate",
            "title": "FrozenLake: Frustration Rate",
            "plot_fn": plot_moving_average_multi,
        },
    ]


def taxi_evaluation_plot_specs() -> list[dict[str, Any]]:
    """Return default evaluation plot specs for Taxi-v3."""
    return [
        {
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "Taxi-v3 Evaluation: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "Taxi-v3 Evaluation: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "episode_won",
            "ylabel": "Success Rate",
            "title": "Taxi-v3 Evaluation: Episode Success",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "taxi_illegal_actions_per_episode",
            "ylabel": "Illegal Actions",
            "title": "Taxi-v3 Evaluation: Illegal Pickup/Drop-off Actions",
            "plot_fn": plot_moving_average_multi,
        },
    ]


def cliffwalking_evaluation_plot_specs() -> list[dict[str, Any]]:
    """Return default evaluation plot specs for CliffWalking."""
    return [
        {
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "CliffWalking Evaluation: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "CliffWalking Evaluation: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "cliff_falls_per_episode",
            "ylabel": "Cliff Falls",
            "title": "CliffWalking Evaluation: Falls Per Episode",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "cliff_fall_rate_per_episode",
            "ylabel": "Cliff Fall Rate",
            "title": "CliffWalking Evaluation: Cliff-Fall Rate",
            "plot_fn": plot_moving_average_multi,
        },
    ]


def frozenlake_evaluation_plot_specs() -> list[dict[str, Any]]:
    """Return default evaluation plot specs for FrozenLake."""
    return [
        {
            "key": "total_reward_per_episode",
            "ylabel": "Return",
            "title": "FrozenLake Evaluation: Episode Return",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "episode_length_per_episode",
            "ylabel": "Episode Length",
            "title": "FrozenLake Evaluation: Episode Length",
            "plot_fn": plot_moving_average_multi,
        },
        {
            "key": "episode_won",
            "ylabel": "Success Rate",
            "title": "FrozenLake Evaluation: Episode Success",
            "plot_fn": plot_moving_average_multi,
        },
    ]
