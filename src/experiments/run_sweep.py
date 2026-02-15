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
