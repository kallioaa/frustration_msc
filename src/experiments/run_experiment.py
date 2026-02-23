"""Minimal experiment runner with JSONL logging."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from evaluation.evaluator import evaluator
from environments.fronzenlake import get_frozenlake_env
from gymnasium.wrappers import TimeLimit


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    name: str = "sarsa_frozenlake"
    num_train_episodes: int = 10000
    env_kwargs: Dict[str, Any] = None
    agent_kwargs: Dict[str, Any] = None


@dataclass
class EvaluateConfig:
    """Configuration for evaluation runs."""

    name: str = "sarsa_frozenlake"
    num_eval_episodes: int = 1000
    seed: int | None = 0
    eval_seeds: list[int] | None = None
    max_episode_steps: int | None = None
    env_kwargs: Dict[str, Any] = None
    evaluation_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None
    td_error_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _concat_metric_logs(metric_runs: list[Dict[str, list[float]]]) -> Dict[str, list[float]]:
    """Concatenate per-run metric logs into a single metric dict."""
    combined: Dict[str, list[float]] = {}
    for metric_dict in metric_runs:
        for metric_name, values in metric_dict.items():
            combined.setdefault(metric_name, []).extend(list(values))
    return combined


def run_training(
    config: TrainingConfig,
    env_factory: Callable[..., Any],
    agent_factory: Callable[..., Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Run a training loop with pluggable env/agent."""
    env_kwargs = config.env_kwargs or {}
    agent_kwargs = config.agent_kwargs or {}
    env = env_factory(**env_kwargs)
    agent = agent_factory(**agent_kwargs)

    train_signature = inspect.signature(agent.train)
    if "eval_env_factory" in train_signature.parameters:
        training_metrics = agent.train(
            env,
            num_episodes=config.num_train_episodes,
            eval_env_factory=env_factory,
            eval_env_kwargs=env_kwargs,
        )
    else:
        training_metrics = agent.train(env, num_episodes=config.num_train_episodes)
    return agent, training_metrics


def run_evaluation(
    config: EvaluateConfig,
    env_factory: Callable[..., Any],
    agent: Any,
) -> Dict[str, Any]:
    """Evaluate a trained agent and return run-level results."""
    env_kwargs = config.env_kwargs or {}
    env = env_factory(**env_kwargs)
    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)

    eval_seeds = (
        [int(seed) for seed in config.eval_seeds]
        if config.eval_seeds is not None
        else [config.seed]
    )
    if not eval_seeds:
        raise ValueError("eval_seeds must be non-empty when provided")

    metric_runs: list[Dict[str, list[float]]] = []
    eval_by_seed: list[Dict[str, Any]] = []
    for eval_seed in eval_seeds:
        eval_metrics = evaluator(
            env,
            agent,
            num_episodes=config.num_eval_episodes,
            seed=eval_seed,
            evaluation_metrics=config.evaluation_metrics,
            td_error_metrics=config.td_error_metrics,
        )
        metric_runs.append(eval_metrics)
        eval_by_seed.append(
            {
                "seed": eval_seed,
                "eval": eval_metrics,
            }
        )

    aggregated_eval_metrics = _concat_metric_logs(metric_runs)

    config_dict = asdict(config)
    evaluation_metrics: Dict[str, Any] = {
        "timestamp": _timestamp(),
        "config": config_dict,
        "eval": aggregated_eval_metrics,
        "eval_by_seed": eval_by_seed,
    }

    return evaluation_metrics
