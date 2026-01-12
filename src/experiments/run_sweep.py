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


def plot_sweep_training_curves(
    results: list[Dict[str, Any]],
    window_size: int = 100,
    label_fn: Callable[[Dict[str, Any]], str] | None = None,
    plot_returns: bool = True,
    plot_episode_won: bool = True,
) -> None:
    """Plot multi-line training curves from sweep results."""
    returns_series: Dict[str, list[float]] = {}
    won_series: Dict[str, list[float]] = {}
    label_counts: Dict[str, int] = {}

    for result in results:
        params = result.get("params", {})
        training = result.get("training", {})
        reward_metrics = training.get("reward", {})
        returns = reward_metrics.get("total_reward_per_episode")
        episode_won = reward_metrics.get("episode_won")

        label = label_fn(params) if label_fn else _format_sweep_label(params)
        label = _unique_label(label, label_counts)

        if plot_returns and returns is not None:
            returns_series[label] = returns
        if plot_episode_won and episode_won is not None:
            won_series[label] = episode_won

    if plot_returns and returns_series:
        plot_moving_average_returns_multi(returns_series, window=window_size)
    if plot_episode_won and won_series:
        plot_moving_average_episode_won_multi(won_series, window=window_size)


def _format_sweep_label(params: Dict[str, Any]) -> str:
    """Create a compact label from agent/env sweep parameters."""
    agent_kwargs = params.get("agent_kwargs") or {}
    env_kwargs = params.get("env_kwargs") or {}
    parts = []

    if agent_kwargs:
        agent_items = ", ".join(f"{k}={v}" for k, v in agent_kwargs.items())
        parts.append(f"agent: {agent_items}")
    if env_kwargs:
        env_items = ", ".join(f"{k}={v}" for k, v in env_kwargs.items())
        parts.append(f"env: {env_items}")

    return " | ".join(parts) if parts else "run"


def _unique_label(label: str, counts: Dict[str, int]) -> str:
    """Ensure labels are unique across series."""
    if label not in counts:
        counts[label] = 1
        return label
    counts[label] += 1
    return f"{label} #{counts[label]}"
