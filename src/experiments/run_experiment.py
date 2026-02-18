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

MAX_EVAL_STEPS = 200


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
    env_kwargs: Dict[str, Any] = None
    evaluation_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None


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
    env = TimeLimit(env, max_episode_steps=MAX_EVAL_STEPS)

    eval_metrics = evaluator(
        env,
        agent,
        num_episodes=config.num_eval_episodes,
        seed=config.seed,
        evaluation_metrics=config.evaluation_metrics,
    )

    config_dict = asdict(config)
    evaluation_metrics: Dict[str, Any] = {
        "timestamp": _timestamp(),
        "config": config_dict,
        "eval": eval_metrics,
    }

    return evaluation_metrics
