from __future__ import annotations

from typing import Callable, Dict, Optional


def evaluator(
    env,
    agent,
    num_episodes: int,
    seed: Optional[int] = None,
    evaluation_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None,
) -> Dict[str, list[float]]:
    """Evaluate agent and return a metrics dict."""
    eval_metric_fns = dict(evaluation_metrics or {})
    if not eval_metric_fns:
        raise ValueError("evaluation_metrics must contain at least one metric")

    metric_logs: Dict[str, list[float]] = {name: [] for name in eval_metric_fns}

    # Seed once for reproducibility (optional)
    if seed is not None:
        env.reset(seed=seed)

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards: list[float] = []

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(float(reward))

        for name, fn in eval_metric_fns.items():
            metric_logs[name].append(float(fn(episode_rewards)))
    return metric_logs
