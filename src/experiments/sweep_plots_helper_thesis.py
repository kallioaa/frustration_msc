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
            {
                "source": "td_error",
                "key": "confirmatory_greedy_positive_ratio_per_episode",
                "ylabel": "Greedy+Positive Ratio",
                "title": "CliffWalking: Confirmatory Greedy+Positive Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "confirmatory_nongreedy_negative_ratio_per_episode",
                "ylabel": "Non-Greedy+Negative Ratio",
                "title": "CliffWalking: Confirmatory Non-Greedy+Negative Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "disconfirmatory_greedy_negative_ratio_per_episode",
                "ylabel": "Greedy+Negative Ratio",
                "title": "CliffWalking: Disconfirmatory Greedy+Negative Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "disconfirmatory_nongreedy_positive_ratio_per_episode",
                "ylabel": "Non-Greedy+Positive Ratio",
                "title": "CliffWalking: Disconfirmatory Non-Greedy+Positive Ratio",
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
            {
                "key": "total_td_error_per_episode",
                "ylabel": "TD Error",
                "title": "CliffWalking Thesis: Evaluation Total TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "mean_absolute_td_error_per_episode",
                "ylabel": "Mean |TD Error|",
                "title": "CliffWalking Thesis: Evaluation Mean Absolute TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "ylabel": "Negative TD Error Sum",
                "title": "CliffWalking Thesis: Evaluation Negative TD Error Sum",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "frustration_rate_per_episode",
                "ylabel": "Frustration Rate",
                "title": "CliffWalking Thesis: Evaluation Frustration Rate",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }


def frozenlake_training_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for FrozenLake training plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "use_td_error_v": True,
        "label_fn": frozenlake_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "episode_won",
                "ylabel": "Success Rate",
                "title": "FrozenLake: Episode Success",
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
                "title": "FrozenLake: Episode Length",
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
                "title": "FrozenLake: Total TD Error",
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
                "title": "FrozenLake: Mean Absolute TD Error",
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
                "title": "FrozenLake: Negative TD Error Sum",
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
                "title": "FrozenLake: Frustration Rate",
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
                "title": "FrozenLake: Confirmatory Update Ratio",
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
                "title": "FrozenLake: Positive-LR Update Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
        ],
    }


def frozenlake_evaluation_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for FrozenLake evaluation plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "label_fn": frozenlake_thesis_label_fn,
        "plot_specs": [
            {
                "key": "episode_won",
                "ylabel": "Success Rate",
                "title": "FrozenLake Thesis: Evaluation Success Rate",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "FrozenLake Thesis: Evaluation Episode Length",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "FrozenLake Thesis: Evaluation Episode Return",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "total_td_error_per_episode",
                "ylabel": "TD Error",
                "title": "FrozenLake Thesis: Evaluation Total TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "mean_absolute_td_error_per_episode",
                "ylabel": "Mean |TD Error|",
                "title": "FrozenLake Thesis: Evaluation Mean Absolute TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "ylabel": "Negative TD Error Sum",
                "title": "FrozenLake Thesis: Evaluation Negative TD Error Sum",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "frustration_rate_per_episode",
                "ylabel": "Frustration Rate",
                "title": "FrozenLake Thesis: Evaluation Frustration Rate",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }


def taxi_training_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for Taxi-v3 training plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "use_td_error_v": True,
        "label_fn": taxi_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "Taxi-v3: Episode Return",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "episode_won",
                "ylabel": "Success Rate",
                "title": "Taxi-v3: Episode Success",
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
                "title": "Taxi-v3: Episode Length",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "taxi_illegal_actions_per_episode",
                "ylabel": "Illegal Actions",
                "title": "Taxi-v3: Illegal Pickup/Drop-off Actions",
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
                "title": "Taxi-v3: Total TD Error",
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
                "title": "Taxi-v3: Mean Absolute TD Error",
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
                "title": "Taxi-v3: Negative TD Error Sum",
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
                "title": "Taxi-v3: Frustration Rate",
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
                "title": "Taxi-v3: Confirmatory Update Ratio",
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
                "title": "Taxi-v3: Positive-LR Update Ratio",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "legend_loc": "best",
                },
            },
        ],
    }


def taxi_evaluation_thesis_config() -> dict[str, Any]:
    """Return thesis-oriented config for Taxi-v3 evaluation plots."""
    return {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "label_fn": taxi_thesis_label_fn,
        "plot_specs": [
            {
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "Taxi-v3 Thesis: Evaluation Episode Return",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "episode_won",
                "ylabel": "Success Rate",
                "title": "Taxi-v3 Thesis: Evaluation Success Rate",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "Taxi-v3 Thesis: Evaluation Episode Length",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "taxi_illegal_actions_per_episode",
                "ylabel": "Illegal Actions",
                "title": "Taxi-v3 Thesis: Evaluation Illegal Actions",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "total_td_error_per_episode",
                "ylabel": "TD Error",
                "title": "Taxi-v3 Thesis: Evaluation Total TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "mean_absolute_td_error_per_episode",
                "ylabel": "Mean |TD Error|",
                "title": "Taxi-v3 Thesis: Evaluation Mean Absolute TD Error",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "ylabel": "Negative TD Error Sum",
                "title": "Taxi-v3 Thesis: Evaluation Negative TD Error Sum",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": (10, 6),
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "frustration_rate_per_episode",
                "ylabel": "Frustration Rate",
                "title": "Taxi-v3 Thesis: Evaluation Frustration Rate",
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


def frozenlake_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for FrozenLake sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}

    alpha_positive = agent_kwargs.get("alpha_positive")
    alpha_negative = agent_kwargs.get("alpha_negative")
    if alpha_positive is not None and alpha_negative not in (None, 0):
        ratio = float(alpha_positive) / float(alpha_negative)
        return f"r={ratio:g}"
    if alpha_positive is not None and alpha_negative is not None:
        return f"ap={alpha_positive}, an={alpha_negative}"

    alpha_conf = agent_kwargs.get("alpha_conf")
    alpha_disconf = agent_kwargs.get("alpha_disconf")
    if alpha_conf is not None and alpha_disconf is not None:
        return f"ac={alpha_conf}, ad={alpha_disconf}"

    return str(agent_kwargs)


def taxi_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for Taxi-v3 sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}

    alpha_positive = agent_kwargs.get("alpha_positive")
    alpha_negative = agent_kwargs.get("alpha_negative")
    if alpha_positive is not None and alpha_negative not in (None, 0):
        ratio = float(alpha_positive) / float(alpha_negative)
        return f"r={ratio:g}"
    if alpha_positive is not None and alpha_negative is not None:
        return f"ap={alpha_positive}, an={alpha_negative}"

    alpha_conf = agent_kwargs.get("alpha_conf")
    alpha_disconf = agent_kwargs.get("alpha_disconf")
    if alpha_conf is not None and alpha_disconf is not None:
        return f"ac={alpha_conf}, ad={alpha_disconf}"

    return str(agent_kwargs)
