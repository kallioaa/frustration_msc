"""Evaluation helpers for experiments."""

from __future__ import annotations

from typing import Optional, Tuple


def evaluate_winrate(
    env, agent, num_episodes: int, seed: Optional[int] = None
) -> Tuple[float, int]:
    """Evaluate agent win rate over episodes."""
    wins = 0

    for _ in range(num_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        if total_reward > 0:
            wins += 1

    win_rate = wins / max(1, num_episodes)
    return win_rate, wins
