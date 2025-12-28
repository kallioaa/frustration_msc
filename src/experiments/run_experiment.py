"""Minimal experiment runner with JSONL logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from agents.sarsa_td0 import SarsaTD0Agent, SarsaTD0Config
from evaluation.evaluator import evaluator
from environments.fronzenlake import get_frozenlake_env
from plots.learning_plots import plot_moving_average_returns
from plots.frustration_plots import (
    plot_moving_average_td_errors,
    plot_frustration_quantiles,
    plot_frustration_rate,
    plot_tail_frustration,
    plot_cvar_tail_frustration,
)


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    name: str = "sarsa_frozenlake"
    num_train_episodes: int = 10000
    seed: int | None = 0
    env_kwargs: Dict[str, Any] = None
    agent_kwargs: Dict[str, Any] = None


@dataclass
class EvaluateConfig:
    """Configuration for evaluation runs."""

    name: str = "sarsa_frozenlake"
    num_eval_episodes: int = 1000
    seed: int | None = 0
    env_kwargs: Dict[str, Any] = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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

    eval_metrics = evaluator(
        env, agent, num_episodes=config.num_eval_episodes, seed=config.seed
    )

    config_dict = asdict(config)
    evaluation_metrics: Dict[str, Any] = {
        "timestamp": _timestamp(),
        "config": config_dict,
        "eval": eval_metrics,
    }

    return evaluation_metrics


def generate_training_plots(
    training_metrics: Dict[str, Any],
    window_size: int = 100,
) -> None:
    """Generate plots from per-episode metrics."""
    total_reward_per_episode = training_metrics.get("reward", {}).get(
        "total_reward_per_episode"
    )
    total_td_error_per_episode = training_metrics.get("td_error", {}).get(
        "total_td_error_per_episode"
    )
    if total_reward_per_episode is not None:
        plot_moving_average_returns(total_reward_per_episode, window=window_size)
    if total_td_error_per_episode is not None:
        plot_moving_average_td_errors(total_td_error_per_episode, window=window_size)
        plot_frustration_quantiles(total_td_error_per_episode, window=window_size)

    frustration_rate_per_episode = training_metrics.get("td_error", {}).get(
        "frustration_rate_per_episode"
    )
    tail_frustration_per_episode = training_metrics.get("td_error", {}).get(
        "tail_frustration_per_episode_90"
    )
    cvar_tail_frustration_per_episode = training_metrics.get("td_error", {}).get(
        "cvar_tail_frustration_per_episode_90"
    )

    if frustration_rate_per_episode is not None:
        plot_frustration_rate(frustration_rate_per_episode, window=window_size)
    if tail_frustration_per_episode is not None:
        plot_tail_frustration(tail_frustration_per_episode, window=window_size)
    if cvar_tail_frustration_per_episode is not None:
        plot_cvar_tail_frustration(cvar_tail_frustration_per_episode, window=window_size)
