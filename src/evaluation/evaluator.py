from __future__ import annotations

from typing import Callable, Dict, Optional


def evaluator(
    env,
    agent,
    num_episodes: int,
    seed: Optional[int] = None,
    evaluation_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None,
    td_error_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None,
) -> Dict[str, list[float]]:
    """Evaluate agent and return a metrics dict."""
    reward_metric_fns = dict(evaluation_metrics or {})
    td_error_metric_fns = dict(td_error_metrics or {})
    if not reward_metric_fns and not td_error_metric_fns:
        raise ValueError(
            "At least one of evaluation_metrics or td_error_metrics must be provided"
        )
    overlap = set(reward_metric_fns).intersection(td_error_metric_fns)
    if overlap:
        overlapping = ", ".join(sorted(overlap))
        raise ValueError(
            f"Metric names overlap between evaluation_metrics and td_error_metrics: {overlapping}"
        )

    metric_logs: Dict[str, list[float]] = {}
    metric_logs.update({name: [] for name in reward_metric_fns})
    metric_logs.update({name: [] for name in td_error_metric_fns})

    if td_error_metric_fns:
        if not hasattr(agent, "q_table") or agent.q_table is None:
            raise ValueError("td_error_metrics require an agent with initialized q_table")
        if not hasattr(agent, "config") or not hasattr(agent.config, "gamma"):
            raise ValueError("td_error_metrics require agent.config.gamma")
        gamma = float(agent.config.gamma)

    # Seed once for reproducibility (optional)
    if seed is not None:
        env.reset(seed=seed)

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards: list[float] = []
        episode_td_errors: list[float] = []

        while not done:
            action = agent.act(state)
            q_sa = float(agent.q_table[state, action]) if td_error_metric_fns else 0.0
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(float(reward))
            if td_error_metric_fns:
                if done:
                    td_target = float(reward)
                else:
                    next_action = agent.act(next_state)
                    td_target = float(reward) + gamma * float(
                        agent.q_table[next_state, next_action]
                    )
                episode_td_errors.append(float(td_target - q_sa))
            state = next_state

        for name, fn in reward_metric_fns.items():
            metric_logs[name].append(float(fn(episode_rewards)))
        for name, fn in td_error_metric_fns.items():
            metric_logs[name].append(float(fn(episode_td_errors)))
    return metric_logs
