"""Utilities for simple parameter sweeps."""

from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Any, Callable, Dict

from experiments.run_experiment import (
    EvaluateConfig,
    TrainingConfig,
    run_evaluation,
    run_training,
)
from plots.reward_plots import (
    plot_moving_average_episode_won_multi,
    plot_moving_average_returns_multi,
)
from plots.evaluation_plots import plot_sweep_win_rate_bar
from plots.td_error_plots import (
    plot_frustration_rate_multi,
    plot_moving_average_td_errors_multi,
    plot_moving_average_td_errors_neg_multi,
)
import numpy as np


def iter_grid(sweep: Dict[str, list[Any]]) -> list[Dict[str, Any]]:
    """Return all combinations for a simple parameter grid."""
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [sweep[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge overrides into a base config, nesting env/agent values under config."""
    merged = dict(base)
    for key in ("env_kwargs", "agent_kwargs"):
        if key not in override:
            continue
        base_config = dict((merged.get(key) or {}).get("config") or {})
        override_config = dict(override.get(key) or {})
        merged[key] = {"config": {**base_config, **override_config}}
    return merged


def run_sweep(
    base_training: TrainingConfig,
    base_evaluation: EvaluateConfig,
    sweep: Dict[str, list[Any]],
    env_factory: Callable[..., Any],
    agent_factory: Callable[..., Any],
) -> list[Dict[str, Any]]:
    """Run a local sweep over env_kwargs and agent_kwargs only."""
    base_training_dict = asdict(base_training)
    base_evaluation_dict = asdict(base_evaluation)
    results: list[Dict[str, Any]] = []

    for override in iter_grid(sweep):
        env_kwargs = override.get("env_kwargs")
        agent_kwargs = override.get("agent_kwargs")

        training_dict = dict(base_training_dict)
        evaluation_dict = dict(base_evaluation_dict)

        if env_kwargs is not None:
            training_dict = merge_config(training_dict, {"env_kwargs": env_kwargs})
            evaluation_dict = merge_config(evaluation_dict, {"env_kwargs": env_kwargs})
        if agent_kwargs is not None:
            training_dict = merge_config(training_dict, {"agent_kwargs": agent_kwargs})

        training_config = TrainingConfig(**training_dict)
        evaluation_config = EvaluateConfig(**evaluation_dict)

        agent, training_metrics = run_training(
            training_config, env_factory=env_factory, agent_factory=agent_factory
        )
        eval_metrics = run_evaluation(
            evaluation_config, env_factory=env_factory, agent=agent
        )

        results.append(
            {
                "params": override,
                "training": training_metrics,
                "evaluation": eval_metrics,
            }
        )

    return results


def plot_sweep_training(
    results: list[Dict[str, Any]],
    window_size: int = 100,
    label_fn: Callable[[Dict[str, Any]], str] | None = None,
    plot_returns: bool = True,
    plot_episode_won: bool = True,
    plot_td_errors: bool = True,
    plot_td_errors_neg_pos: bool = True,
    plot_frustration_rate: bool = True,
) -> None:
    """Plot multi-line training curves from sweep results."""
    returns_series: Dict[str, list[float]] = {}
    won_series: Dict[str, list[float]] = {}
    td_error_series: Dict[str, list[float]] = {}
    frustration_rate_series: Dict[str, list[float]] = {}

    grouped: Dict[tuple, Dict[str, Any]] = {}

    for result in results:
        params = result.get("params", {})
        grouped_params = _strip_seed(params)
        key = _freeze(grouped_params)

        training = result.get("training", {})
        reward_metrics = training.get("reward", {})
        td_error_metrics = training.get("td_error", {})

        entry = grouped.setdefault(
            key,
            {
                "params": grouped_params,
                "returns": [],
                "episode_won": [],
                "td_error": [],
                "frustration_rate": [],
            },
        )

        returns = reward_metrics.get("total_reward_per_episode")
        episode_won = reward_metrics.get("episode_won")
        total_td_error = td_error_metrics.get("total_td_error_per_episode")
        frustration_rate = td_error_metrics.get("frustration_rate_per_episode")

        if plot_returns and returns is not None:
            entry["returns"].append(returns)
        if plot_episode_won and episode_won is not None:
            entry["episode_won"].append(episode_won)
        if plot_td_errors and total_td_error is not None:
            entry["td_error"].append(total_td_error)
        if plot_frustration_rate and frustration_rate is not None:
            entry["frustration_rate"].append(frustration_rate)

    for entry in grouped.values():
        params = entry["params"]
        label = label_fn(params) if label_fn else _format_sweep_label(params)

        if plot_returns and entry["returns"]:
            returns_series[label] = _mean_series(entry["returns"])
        if plot_episode_won and entry["episode_won"]:
            won_series[label] = _mean_series(entry["episode_won"])
        if plot_td_errors and entry["td_error"]:
            td_error_series[label] = _mean_series(entry["td_error"])
        if plot_frustration_rate and entry["frustration_rate"]:
            frustration_rate_series[label] = _mean_series(entry["frustration_rate"])

    if plot_returns and returns_series:
        plot_moving_average_returns_multi(returns_series, window=window_size)
    if plot_episode_won and won_series:
        plot_moving_average_episode_won_multi(won_series, window=window_size)
    if plot_td_errors and td_error_series:
        plot_moving_average_td_errors_multi(td_error_series, window=window_size)
    if plot_td_errors_neg_pos and td_error_series:
        plot_moving_average_td_errors_neg_multi(td_error_series, window=window_size)
    if plot_frustration_rate and frustration_rate_series:
        plot_frustration_rate_multi(frustration_rate_series, window=window_size)


def plot_sweep_evaluation(
    results: list[Dict[str, Any]],
    label_fn: Callable[[Dict[str, Any]], str] | None = None,
) -> None:
    """Plot evaluation summaries for sweep runs."""
    grouped: Dict[tuple, Dict[str, Any]] = {}

    for result in results:
        params = result.get("params", {})
        grouped_params = _strip_seed(params)
        key = _freeze(grouped_params)

        evaluation = result.get("evaluation", {}).get("eval", {})
        win_rate = evaluation.get("win_rate")
        if win_rate is None:
            continue

        entry = grouped.setdefault(
            key, {"params": grouped_params, "win_rates": []}
        )
        entry["win_rates"].append(float(win_rate))

    labels: list[str] = []
    values: list[float] = []
    for entry in grouped.values():
        params = entry["params"]
        label = label_fn(params) if label_fn else _format_sweep_label(params)
        labels.append(label)
        values.append(float(np.mean(entry["win_rates"])))

    plot_sweep_win_rate_bar(labels, values)


def _format_sweep_label(params: Dict[str, Any]) -> str:
    """Create a compact label from agent/env sweep parameters."""
    agent_kwargs = params.get("agent_kwargs") or {}
    env_kwargs = params.get("env_kwargs") or {}
    parts = []

    if agent_kwargs:
        q_table_label = agent_kwargs.get("initial_q_table_label")
        if q_table_label is None and "initial_q_table" in agent_kwargs:
            q_table_label = "zeros"
        agent_items_list = []
        for key, value in agent_kwargs.items():
            if key == "initial_q_table":
                continue
            if key == "initial_q_table_label":
                agent_items_list.append(f"q_table={value}")
                continue
            agent_items_list.append(f"{key}={value}")
        if q_table_label is not None and "initial_q_table_label" not in agent_kwargs:
            agent_items_list.append(f"q_table={q_table_label}")
        agent_items = ", ".join(agent_items_list)
        parts.append(f"agent: {agent_items}")
    if env_kwargs:
        env_items = ", ".join(f"{k}={v}" for k, v in env_kwargs.items())
        parts.append(f"env: {env_items}")

    return " | ".join(parts) if parts else "run"


def _strip_seed(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return params without seed keys for grouping/labeling."""
    agent_kwargs = dict(params.get("agent_kwargs") or {})
    env_kwargs = dict(params.get("env_kwargs") or {})

    agent_kwargs.pop("seed", None)
    env_kwargs.pop("seed", None)

    stripped: Dict[str, Any] = {}
    if agent_kwargs:
        stripped["agent_kwargs"] = agent_kwargs
    if env_kwargs:
        stripped["env_kwargs"] = env_kwargs
    return stripped


def _freeze(value: Any) -> Any:
    """Make a nested structure hashable for dict keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mean_series(series_list: list[list[float]]) -> list[float]:
    """Element-wise mean of multiple series, truncating to shortest."""
    min_len = min(len(series) for series in series_list)
    if min_len == 0:
        return []
    stacked = np.array([series[:min_len] for series in series_list], dtype=float)
    return list(np.mean(stacked, axis=0))
