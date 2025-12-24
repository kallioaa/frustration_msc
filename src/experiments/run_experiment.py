"""Minimal experiment runner with JSONL logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np

from agents.sarsa_td0 import SarsaTD0Agent, SarsaTD0Config
from environments.fronzenlake import get_frozenlake_env
from metrics.evaluate_win_rate import evaluate_win_rate
from plots.plot_episode_returns import plot_episode_returns


@dataclass
class ExperimentConfig:
    """Configuration for a single training + evaluation run."""

    name: str = "sarsa_frozenlake"
    num_train_episodes: int = 10000
    num_eval_episodes: int = 1000
    seed: int | None = 0
    env_kwargs: Dict[str, Any] = None
    agent_kwargs: Dict[str, Any] = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_experiment(
    config: ExperimentConfig,
    env_factory: Callable[..., Any],
    agent_factory: Callable[..., Any],
    evaluator: Callable[..., Tuple[float, int]] = evaluate_win_rate,
) -> Dict[str, Any]:
    """Run a training + evaluation loop with pluggable env/agent."""
    env_kwargs = config.env_kwargs or {}
    agent_kwargs = config.agent_kwargs or {}
    env = env_factory(**env_kwargs)
    agent = agent_factory(**agent_kwargs)

    returns, td_errors = agent.train(env, num_episodes=config.num_train_episodes)
    win_rate, total_wins = evaluator(
        env, agent, num_episodes=config.num_eval_episodes, seed=config.seed
    )

    result = {
        "timestamp": _timestamp(),
        "config": asdict(config),
        "train": {
            "episodes": config.num_train_episodes,
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "final_return": float(returns[-1]) if returns else 0.0,
            "td_error_mean": float(np.mean(td_errors)) if td_errors else 0.0,
        },
        "eval": {
            "episodes": config.num_eval_episodes,
            "win_rate": float(win_rate),
            "wins": int(total_wins),
        },
    }
    return result


def save_run(
    result: Dict[str, Any],
    runs_dir: Path = Path("runs"),
    save_plots: bool = False,
    returns: list[float] | None = None,
) -> Path:
    """Save a run to JSONL and optionally write plot artifacts."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = result["timestamp"]
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Append to a JSONL log for easy aggregation.
    log_path = runs_dir / "runs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    # Save config and optionally a returns plot.
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if save_plots and returns is not None:
        plot_episode_returns(returns)

    return run_dir


def main() -> Tuple[Dict[str, Any], Path]:
    """Example entrypoint; adjust config as needed."""
    config = ExperimentConfig(
        name="sarsa_frozenlake",
        num_train_episodes=10000,
        num_eval_episodes=1000,
        seed=0,
        env_kwargs={"map_name": "4x4", "is_slippery": False},
        agent_kwargs={
            "config": SarsaTD0Config(alpha=0.1, gamma=0.99, epsilon=0.1, seed=0)
        },
    )

    result = run_experiment(
        config,
        env_factory=get_frozenlake_env,
        agent_factory=SarsaTD0Agent,
    )
    run_dir = save_run(result, save_plots=False)
    return result, run_dir


if __name__ == "__main__":
    main()
