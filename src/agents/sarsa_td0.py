"""Simple SARSA (TD(0)) agent for discrete environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SarsaTD0Config:
    """Hyperparameters for SARSA (TD(0))."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: Optional[int] = None


class SarsaTD0Agent:
    """Tabular SARSA (TD(0)) for discrete state/action spaces."""

    def __init__(self, config: SarsaTD0Config):
        self.config = config
        self.q_table: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(config.seed)

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        # Choose random action with probability epsilon, otherwise greedy.
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def train(self, env, num_episodes: int) -> List[float]:
        """Train SARSA (TD(0)) and return per-episode rewards."""
        rewards: List[float] = []
        td_errors: List[float] = []

        epsilon = self.config.epsilon

        # Build a fresh Q-table from env's discrete spaces.
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        for _ in range(num_episodes):
            state, _ = env.reset(seed=self.config.seed)
            action = self._epsilon_greedy(state, epsilon)
            done = False
            total_reward = 0.0

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = (
                    self._epsilon_greedy(next_state, epsilon) if not done else 0
                )

                # SARSA update: bootstrap from next action's value.
                td_target = reward + self.config.gamma * self.q_table[
                    next_state, next_action
                ] * (0.0 if done else 1.0)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.config.alpha * td_error

                state, action = next_state, next_action
                total_reward += reward
                td_errors.append(td_error)

        return rewards, td_errors

    def act(self, state: int) -> int:
        """Greedy action from current Q-table."""
        return int(np.argmax(self.q_table[state]))
