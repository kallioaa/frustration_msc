"""Thesis-specific sweep plot configs."""

from __future__ import annotations

from typing import Any

from plots.plots import plot_bar_mean_multi, plot_moving_average_multi


def cliffwalking_training_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for CliffWalking training plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "use_td_error_v": True,
        "label_fn": cliffwalking_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "CliffWalking: Episode Length",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "cliff_falls_per_episode",
                "ylabel": "Cliff Falls",
                "title": "CliffWalking: Falls Per Episode",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "total_td_error_per_episode",
                "ylabel": "TD Error",
                "title": "CliffWalking: Total TD Error",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "mean_absolute_td_error_per_episode",
                "ylabel": "Mean |TD Error|",
                "title": "CliffWalking: Mean Absolute TD Error",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "negative_td_error_sum_per_episode",
                "ylabel": "Negative TD Error Sum",
                "title": "CliffWalking: Negative TD Error Sum",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "frustration_rate_per_episode",
                "ylabel": "Frustration Rate",
                "title": "CliffWalking: Frustration Rate",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "confirmatory_update_ratio_per_episode",
                "ylabel": "Confirmatory Update Ratio",
                "title": "CliffWalking: Confirmatory Update Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "positive_lr_update_ratio_per_episode",
                "ylabel": "Positive-LR Update Ratio",
                "title": "CliffWalking: Positive-LR Update Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
        ],
    }


def cliffwalking_evaluation_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for CliffWalking evaluation plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "label_fn": cliffwalking_thesis_label_fn,
        "plot_specs": [
            {
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "CliffWalking Thesis: Evaluation Episode Length",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }


def cliffwalking_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for CliffWalking sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}

    alpha_conf = agent_kwargs.get("alpha_conf")
    alpha_disconf = agent_kwargs.get("alpha_disconf")
    if alpha_conf is not None and alpha_disconf is not None:
        return f"ac={alpha_conf}, ad={alpha_disconf}"

    alpha_positive = agent_kwargs.get("alpha_positive")
    alpha_negative = agent_kwargs.get("alpha_negative")
    if alpha_positive is not None and alpha_negative is not None:
        return f"ap={alpha_positive}, an={alpha_negative}"

    return str(agent_kwargs)
