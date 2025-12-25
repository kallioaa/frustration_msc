from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SarsaTD0Config:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    seed: Optional[int] = None


class SarsaTD0Agent:
    def __init__(self, config: SarsaTD0Config):
        self.config = config
        self.q_table: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(config.seed)

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def train(self, env, num_episodes: int) -> Tuple[List[float], List[float]]:
        rewards: List[float] = []
        td_errors: List[float] = []

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        # Seed once for reproducibility (optional)
        if self.config.seed is not None:
            env.reset(seed=self.config.seed)

        epsilon = self.config.epsilon

        for _ in range(num_episodes):
            state, _info = env.reset()  # don't reseed every episode
            action = self._epsilon_greedy(state, epsilon)

            total_reward = 0.0
            total_td_errors = 0.0
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

                if done:
                    td_target = float(reward)
                else:
                    next_action = self._epsilon_greedy(next_state, epsilon)
                    td_target = (
                        float(reward)
                        + self.config.gamma * self.q_table[next_state, next_action]
                    )

                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.config.alpha * td_error

                # append td_errors
                total_td_errors += td_error

                if not done:
                    state, action = next_state, next_action

            rewards.append(total_reward)
            epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
            td_errors.append(total_td_errors)

        return rewards, td_errors

    def act(self, state: int) -> int:
        if self.q_table is None:
            raise RuntimeError("Q-table is not initialized. Call train() first.")
        return int(np.argmax(self.q_table[state]))
