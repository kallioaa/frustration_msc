from typing import Dict, Optional


def evaluator(
    env, agent, num_episodes: int, seed: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate agent and return a metrics dict."""
    wins = 0

    # Seed once for reproducibility (optional)
    if seed is not None:
        env.reset(seed=seed)

    for _ in range(num_episodes):
        state, _ = env.reset()
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

    evaluation_metrics = {
        "win_rate": float(win_rate),
        "wins": float(wins),
        "episodes": float(num_episodes),
    }

    return evaluation_metrics
