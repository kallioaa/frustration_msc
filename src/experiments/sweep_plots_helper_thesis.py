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
        "use_td_error_v": False,
        "label_fn": cliffwalking_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "CliffWalking Thesis: Training Return",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "CliffWalking Thesis: Training Episode Length",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "cliff_fall_rate_per_episode",
                "ylabel": "Cliff Fall Rate",
                "title": "CliffWalking Thesis: Training Cliff-Fall Rate",
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
                "title": "CliffWalking Thesis: Training Frustration Rate",
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
