from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class SarsaExpectedConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: Optional[int] = None
    reward_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None
    td_error_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None


class SarsaExpectedAgent:
    def __init__(self, config: SarsaExpectedConfig):
        self.config = config
        self.q_table: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(config.seed)

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def _expected_q(self, state: int, epsilon: float) -> float:
        q_values = self.q_table[state]
        n_actions = q_values.shape[0]
        max_value = np.max(q_values)
        greedy_actions = np.flatnonzero(q_values == max_value)

        probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
        probs[greedy_actions] += (1.0 - epsilon) / len(greedy_actions)
        return float(np.dot(probs, q_values))

    def train(self, env, num_episodes: int) -> Dict[str, Dict[str, List[float]]]:
        reward_metrics = self.config.reward_metrics or {}
        td_error_metrics = self.config.td_error_metrics or {}
        reward_metrics_log: Dict[str, List[float]] = {
            name: [] for name in reward_metrics
        }
        td_error_metrics_log: Dict[str, List[float]] = {
            name: [] for name in td_error_metrics
        }

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        if self.config.seed is not None:
            env.reset(seed=self.config.seed)

        epsilon = self.config.epsilon

        for _ in range(num_episodes):
            episode_rewards: List[float] = []
            episode_td_errors: List[float] = []

            state, _info = env.reset()
            action = self._epsilon_greedy(state, epsilon)
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if done:
                    td_target = float(reward)
                else:
                    expected_q = self._expected_q(next_state, epsilon)
                    td_target = float(reward) + self.config.gamma * expected_q

                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.config.alpha * td_error

                if not done:
                    action = self._epsilon_greedy(next_state, epsilon)
                    state = next_state

                episode_rewards.append(float(reward))
                episode_td_errors.append(float(td_error))

            for name, fn in reward_metrics.items():
                reward_metrics_log[name].append(float(fn(episode_rewards)))
            for name, fn in td_error_metrics.items():
                td_error_metrics_log[name].append(float(fn(episode_td_errors)))

        episode_metrics: Dict[str, Dict[str, List[float]]] = {
            "reward": reward_metrics_log,
            "td_error": td_error_metrics_log,
        }
        return episode_metrics

    def act(self, state: int) -> int:
        if self.q_table is None:
            raise RuntimeError("Q-table is not initialized. Call train() first.")
        return int(np.argmax(self.q_table[state]))
