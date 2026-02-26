"""Thesis-specific sweep plot configs."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from plots.plots import plot_bar_mean_multi, plot_moving_average_multi

THESIS_FIGSIZE = (5.9, 3.32)


def _apply_bias_gradient_color_toggle(
    config: dict[str, Any],
    enabled: bool,
) -> None:
    """Write the bias-gradient color toggle into every plot spec's plot_kwargs."""
    config["enable_bias_gradient_colors"] = bool(enabled)
    for spec in config.get("plot_specs", []):
        plot_kwargs = dict(spec.get("plot_kwargs") or {})
        plot_kwargs["enable_bias_gradient_colors"] = bool(enabled)
        spec["plot_kwargs"] = plot_kwargs


def _apply_bar_plot_order(
    config: dict[str, Any],
    bar_order: str,
) -> None:
    """Write bar ordering into bar-plot specs only."""
    config["bar_order"] = bar_order
    for spec in config.get("plot_specs", []):
        if spec.get("plot_fn") is not plot_bar_mean_multi:
            continue
        plot_kwargs = dict(spec.get("plot_kwargs") or {})
        plot_kwargs["bar_order"] = bar_order
        spec["plot_kwargs"] = plot_kwargs


def _spec_override_key(spec: dict[str, Any]) -> str | None:
    """Return the key used for per-plot override maps."""
    override_key = spec.get("override_key")
    if override_key is not None:
        return override_key
    key = spec.get("key")
    return key if isinstance(key, str) else None


def _with_config_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deep-copied plot config with top-level and per-metric overrides."""
    updated = deepcopy(config)
    _apply_bias_gradient_color_toggle(updated, enabled=False)
    _apply_bar_plot_order(updated, bar_order="original")

    if not overrides:
        return updated

    override_dict = dict(overrides)
    ylims_by_key = override_dict.pop("ylims_by_key", None)
    plot_kwargs_by_key = override_dict.pop("plot_kwargs_by_key", None)
    enable_bias_gradient_colors = override_dict.pop("enable_bias_gradient_colors", None)
    bar_order = override_dict.pop("bar_order", None)
    evaluation_bar_order = override_dict.pop("evaluation_bar_order", None)

    for key, value in override_dict.items():
        updated[key] = value

    if enable_bias_gradient_colors is not None:
        _apply_bias_gradient_color_toggle(updated, enabled=bool(enable_bias_gradient_colors))
    resolved_bar_order = (
        evaluation_bar_order if evaluation_bar_order is not None else bar_order
    )
    if resolved_bar_order is not None:
        _apply_bar_plot_order(updated, bar_order=str(resolved_bar_order))

    if ylims_by_key:
        for spec in updated.get("plot_specs", []):
            key = _spec_override_key(spec)
            if key not in ylims_by_key:
                continue
            plot_kwargs = dict(spec.get("plot_kwargs") or {})
            ylim = ylims_by_key[key]
            if ylim is None:
                plot_kwargs.pop("ylim", None)
            else:
                plot_kwargs["ylim"] = ylim
            spec["plot_kwargs"] = plot_kwargs

    if plot_kwargs_by_key:
        for spec in updated.get("plot_specs", []):
            metric_key = _spec_override_key(spec)
            metric_overrides = plot_kwargs_by_key.get(metric_key)
            if not metric_overrides:
                continue
            plot_kwargs = dict(spec.get("plot_kwargs") or {})
            plot_kwargs.update(metric_overrides)
            spec["plot_kwargs"] = plot_kwargs

    return updated


def cliffwalking_training_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for CliffWalking training plots."""
    config = {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "use_td_error_v": True,
        "label_fn": cliffwalking_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "CliffWalking: Episode Return",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "CliffWalking: Episode Length",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "CliffWalking: Reward Loss",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "effective_learning_rate_per_episode",
                "ylabel": "Effective Learning Rate",
                "title": "CliffWalking: Effective Learning Rate",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def cliffwalking_evaluation_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for CliffWalking evaluation plots."""
    config = {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "label_fn": cliffwalking_thesis_label_fn,
        "plot_specs": [
            {
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "CliffWalking Thesis: Evaluation Episode Return",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "episode_length_per_episode",
                "ylabel": "Episode Length",
                "title": "CliffWalking Thesis: Evaluation Episode Length",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "CliffWalking Thesis: Evaluation Reward Loss",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "cliff_falls_per_episode",
                "ylabel": "Cliff Falls",
                "title": "CliffWalking Thesis: Evaluation Falls Per Episode",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def frozenlake_training_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for FrozenLake training plots."""
    config = {
        "window_size": 100,
        "start_episode": 0,
        "end_episode": None,
        "use_td_error_v": True,
        "label_fn": frozenlake_thesis_label_fn,
        "plot_specs": [
            {
                "source": "reward",
                "key": "total_reward_per_episode",
                "ylabel": "Return",
                "title": "FrozenLake: Episode Return",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "reward",
                "key": "episode_won",
                "ylabel": "Success Rate",
                "title": "FrozenLake: Episode Success",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "FrozenLake: Reward Loss",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "effective_learning_rate_per_episode",
                "ylabel": "Effective Learning Rate",
                "title": "FrozenLake: Effective Learning Rate",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def frozenlake_evaluation_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for FrozenLake evaluation plots."""
    config = {
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "FrozenLake Thesis: Evaluation Reward Loss",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def taxi_training_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for Taxi-v3 training plots."""
    config = {
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "Taxi-v3: Reward Loss",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
            {
                "source": "td_error",
                "key": "effective_learning_rate_per_episode",
                "ylabel": "Effective Learning Rate",
                "title": "Taxi-v3: Effective Learning Rate",
                "plot_fn": plot_moving_average_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
                    "legend_loc": "best",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def taxi_evaluation_thesis_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return thesis-oriented config for Taxi-v3 evaluation plots."""
    config = {
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
            {
                "key": "negative_td_error_sum_per_episode",
                "metric_transform": "abs",
                "override_key": "reward_loss_per_episode",
                "ylabel": "Reward Loss",
                "title": "Taxi-v3 Thesis: Evaluation Reward Loss",
                "plot_fn": plot_bar_mean_multi,
                "plot_kwargs": {
                    "figsize": THESIS_FIGSIZE,
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
                    "figsize": THESIS_FIGSIZE,
                    "xtick_rotation": 45,
                    "xtick_ha": "right",
                },
            },
        ],
    }
    return _with_config_overrides(config, overrides)


def _thesis_bias_lr_label(agent_kwargs: dict[str, Any]) -> str:
    """Return compact labels, marking equal-rate settings as a shared baseline."""
    alpha_conf = agent_kwargs.get("alpha_conf")
    alpha_disconf = agent_kwargs.get("alpha_disconf")
    if alpha_conf is not None and alpha_disconf is not None:
        ac = float(alpha_conf)
        ad = float(alpha_disconf)
        if math.isclose(ac, ad, rel_tol=0.0, abs_tol=1e-12):
            return f"a={ac:.3f}"
        return f"ac={ac:.3f}, ad={ad:.3f}"

    alpha_positive = agent_kwargs.get("alpha_positive")
    alpha_negative = agent_kwargs.get("alpha_negative")
    if alpha_positive is not None and alpha_negative is not None:
        ap = float(alpha_positive)
        an = float(alpha_negative)
        if math.isclose(ap, an, rel_tol=0.0, abs_tol=1e-12):
            return f"a={ap:.3f}"
        return f"ap={ap:.3f}, an={an:.3f}"

    return str(agent_kwargs)


def cliffwalking_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for CliffWalking sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}
    return _thesis_bias_lr_label(agent_kwargs)


def frozenlake_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for FrozenLake sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}
    return _thesis_bias_lr_label(agent_kwargs)


def taxi_thesis_label_fn(params: dict[str, Any]) -> str:
    """Compact, thesis-friendly labels for Taxi-v3 sweeps."""
    agent_kwargs = params.get("agent_kwargs") or {}
    return _thesis_bias_lr_label(agent_kwargs)
