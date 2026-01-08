from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class SarsaTD0EntropicConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1

    # Entropic risk parameter (must be nonzero)
    beta: float = 0.1

    # Numerics / stability
    exp_clip: float = 700.0  # clips beta*G before exp to avoid overflow
    u_floor: float = 1e-300  # ensures log(U) is well-defined

    seed: Optional[int] = None
    reward_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None
    td_error_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None


class SarsaTD0EntropicAgent:
    """
    Tabular on-policy SARSA(0) for the entropic risk measure.

    Maintains U(s,a) ≈ E[exp(beta * G)] and derives Q(s,a) = (1/beta) * log U(s,a).
    Update:
        U(s,a) <- U(s,a) + alpha * ( exp(beta * G) - U(s,a) )
    where:
        G = r + gamma * Q(s',a')   (or G = r at terminal)
    """

    def __init__(self, config: SarsaTD0EntropicConfig):
        if abs(config.beta) < 1e-12:
            raise ValueError(
                "SarsaTD0EntropicAgent requires beta != 0. Put risk-neutral SARSA in a separate class."
            )
        self.config = config

        self.u_table: Optional[np.ndarray] = None  # stores U(s,a)
        self.q_table: Optional[np.ndarray] = (
            None  # optional cached Q(s,a) after training
        )
        self._rng = np.random.default_rng(config.seed)

    def _greedy_action(self, state: int) -> int:
        """
        Greedy w.r.t Q(s,a) = log(U)/beta.
        Since log is monotone:
          - beta > 0: argmax Q == argmax U
          - beta < 0: argmax Q == argmin U   (division by negative flips ordering)
        """
        assert self.u_table is not None
        if self.config.beta > 0:
            return int(np.argmax(self.u_table[state]))
        return int(np.argmin(self.u_table[state]))

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        assert self.u_table is not None
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.u_table.shape[1]))
        return self._greedy_action(state)

    def _q_from_u(self, u: float) -> float:
        # Safe log
        u_safe = max(u, self.config.u_floor)
        return float(np.log(u_safe) / self.config.beta)

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

        # Initialize U(s,a). If Q starts at 0, then U = exp(beta*0) = 1.
        self.u_table = np.ones((n_states, n_actions), dtype=np.float64)
        self.q_table = None  # will be filled at end (optional cache)

        if self.config.seed is not None:
            env.reset(seed=self.config.seed)

        epsilon = self.config.epsilon
        beta = self.config.beta

        for _ in range(num_episodes):
            state, _info = env.reset()
            action = self._epsilon_greedy(state, epsilon)

            episode_rewards: List[float] = []
            episode_td_errors_q: List[float] = []

            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Q(s,a) before update (for logging)
                u_sa = float(self.u_table[state, action])
                q_sa_before = self._q_from_u(u_sa)

                if done:
                    G = float(reward)
                else:
                    next_action = self._epsilon_greedy(next_state, epsilon)
                    u_next = float(self.u_table[next_state, next_action])
                    q_next = self._q_from_u(u_next)
                    G = float(reward) + self.config.gamma * q_next

                # U-target: exp(beta * G), with clipping for numerical stability
                x = float(
                    np.clip(beta * G, -self.config.exp_clip, self.config.exp_clip)
                )
                u_target = float(np.exp(x))

                # TD(0) update in U-space
                u_new = u_sa + self.config.alpha * (u_target - u_sa)
                # keep strictly positive for log
                if u_new < self.config.u_floor:
                    u_new = self.config.u_floor
                self.u_table[state, action] = u_new

                # For metrics: Q-target y = (1/beta) log u_target = x/beta (equals G unless clipped)
                y_q = float(x / beta)
                td_error_q = y_q - q_sa_before

                episode_rewards.append(float(reward))
                episode_td_errors_q.append(float(td_error_q))

                if not done:
                    state, action = next_state, next_action

            for name, fn in reward_metrics.items():
                reward_metrics_log[name].append(float(fn(episode_rewards)))
            for name, fn in td_error_metrics.items():
                td_error_metrics_log[name].append(float(fn(episode_td_errors_q)))

        # Optional: cache Q-table at end (useful for act()).
        self.q_table = np.log(np.maximum(self.u_table, self.config.u_floor)) / beta

        return {"reward": reward_metrics_log, "td_error": td_error_metrics_log}

    def act(self, state: int) -> int:
        if self.u_table is None:
            raise RuntimeError("U-table is not initialized. Call train() first.")
        return self._greedy_action(state)
