from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SarsaExpectedConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: Optional[int] = None


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

    def train(self, env, num_episodes: int) -> Tuple[List[float], List[float]]:
        rewards: List[float] = []
        td_errors: List[float] = []

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        if self.config.seed is not None:
            env.reset(seed=self.config.seed)

        epsilon = self.config.epsilon

        for _ in range(num_episodes):
            state, _info = env.reset()
            action = self._epsilon_greedy(state, epsilon)

            total_reward = 0.0
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

                if done:
                    td_target = float(reward)
                else:
                    expected_q = self._expected_q(next_state, epsilon)
                    td_target = float(reward) + self.config.gamma * expected_q

                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.config.alpha * td_error
                td_errors.append(float(td_error))

                if not done:
                    action = self._epsilon_greedy(next_state, epsilon)
                    state = next_state

            rewards.append(total_reward)

        return rewards, td_errors

    def act(self, state: int) -> int:
        if self.q_table is None:
            raise RuntimeError("Q-table is not initialized. Call train() first.")
        return int(np.argmax(self.q_table[state]))
