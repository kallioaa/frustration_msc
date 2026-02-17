from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class SarsaTD0ConfirmationBiasConfig:
    alpha_conf: float = 0.1
    alpha_disconf: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.1
    initial_q_table: Optional[np.ndarray] = None
    initial_q_table_label: str = "zeros"
    seed: Optional[int] = None
    reward_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None
    td_error_metrics: Optional[Dict[str, Callable[[list[float]], float]]] = None


class SarsaTD0ConfirmationBiasAgent:
    def __init__(self, config: SarsaTD0ConfirmationBiasConfig | dict):
        if isinstance(config, dict):
            config = SarsaTD0ConfirmationBiasConfig(**config)
        self.config = config
        self.q_table: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(config.seed)

    def _epsilon_greedy(self, state: int, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def _v_from_q_eps_greedy(self, state: int) -> float:
        epsilon = self.config.epsilon
        q_s = self.q_table[state]
        n_actions = q_s.shape[0]

        max_q = np.max(q_s)
        greedy_actions = np.flatnonzero(q_s == max_q)
        n_greedy = greedy_actions.size

        p = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
        p[greedy_actions] += (1.0 - epsilon) / n_greedy

        return float(np.dot(p, q_s))

    def train(self, env, num_episodes: int) -> Dict[str, Dict[str, List[float]]]:
        reward_metrics = self.config.reward_metrics or {}
        td_error_metrics = self.config.td_error_metrics or {}
        reward_metrics_log: Dict[str, List[float]] = {
            name: [] for name in reward_metrics
        }
        td_error_metrics_log: Dict[str, List[float]] = {}
        for name in td_error_metrics:
            td_error_metrics_log[name] = []
            td_error_metrics_log[f"{name}_v"] = []
        td_error_metrics_log["confirmatory_update_ratio_per_episode"] = []
        td_error_metrics_log["confirmatory_greedy_positive_ratio_per_episode"] = []
        td_error_metrics_log["confirmatory_nongreedy_negative_ratio_per_episode"] = []
        td_error_metrics_log["disconfirmatory_greedy_negative_ratio_per_episode"] = []
        td_error_metrics_log["disconfirmatory_nongreedy_positive_ratio_per_episode"] = []

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        if self.config.initial_q_table is None:
            self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        else:
            q_table = np.asarray(self.config.initial_q_table, dtype=np.float64)
            if q_table.shape != (n_states, n_actions):
                raise ValueError(
                    "initial_q_table must match (num_states, num_actions)."
                )
            self.q_table = q_table.copy()

        if self.config.seed is not None:
            env.reset(seed=self.config.seed)

        epsilon = self.config.epsilon

        for _ in range(num_episodes):
            state, _info = env.reset()
            action = self._epsilon_greedy(state, epsilon)
            episode_rewards: List[float] = []
            episode_td_errors: List[float] = []
            episode_td_errors_v: List[float] = []
            episode_confirmatory_updates = 0
            episode_disconfirmatory_updates = 0
            episode_confirmatory_greedy_positive_updates = 0
            episode_confirmatory_nongreedy_negative_updates = 0
            episode_disconfirmatory_greedy_negative_updates = 0
            episode_disconfirmatory_nongreedy_positive_updates = 0
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if done:
                    td_target = float(reward)
                else:
                    next_action = self._epsilon_greedy(next_state, epsilon)
                    td_target = (
                        float(reward)
                        + self.config.gamma * self.q_table[next_state, next_action]
                    )

                td_error = td_target - self.q_table[state, action]
                a_star = int(np.argmax(self.q_table[state]))
                if action == a_star and td_error > 0.0:
                    episode_confirmatory_greedy_positive_updates += 1
                elif action != a_star and td_error < 0.0:
                    episode_confirmatory_nongreedy_negative_updates += 1
                elif action == a_star and td_error < 0.0:
                    episode_disconfirmatory_greedy_negative_updates += 1
                elif action != a_star and td_error > 0.0:
                    episode_disconfirmatory_nongreedy_positive_updates += 1
                confirm = (action == a_star and td_error > 0.0) or (
                    action != a_star and td_error < 0.0
                )
                alpha = self.config.alpha_conf if confirm else self.config.alpha_disconf
                if confirm:
                    episode_confirmatory_updates += 1
                else:
                    episode_disconfirmatory_updates += 1
                self.q_table[state, action] += alpha * td_error

                if done:
                    v_next = 0.0
                else:
                    v_next = self._v_from_q_eps_greedy(next_state)

                v_state = self._v_from_q_eps_greedy(state)
                td_error_v = float(reward) + self.config.gamma * v_next - v_state

                episode_rewards.append(float(reward))
                episode_td_errors.append(float(td_error))
                episode_td_errors_v.append(float(td_error_v))

                if not done:
                    state, action = next_state, next_action

            for name, fn in reward_metrics.items():
                reward_metrics_log[name].append(float(fn(episode_rewards)))
            for name, fn in td_error_metrics.items():
                td_error_metrics_log[name].append(float(fn(episode_td_errors)))
                td_error_metrics_log[f"{name}_v"].append(float(fn(episode_td_errors_v)))
            total_updates = (
                episode_confirmatory_updates + episode_disconfirmatory_updates
            )
            confirmatory_ratio = (
                (episode_confirmatory_updates / total_updates)
                if total_updates > 0
                else 0.0
            )
            confirmatory_greedy_positive_ratio = (
                (episode_confirmatory_greedy_positive_updates / total_updates)
                if total_updates > 0
                else 0.0
            )
            confirmatory_nongreedy_negative_ratio = (
                (episode_confirmatory_nongreedy_negative_updates / total_updates)
                if total_updates > 0
                else 0.0
            )
            disconfirmatory_greedy_negative_ratio = (
                (episode_disconfirmatory_greedy_negative_updates / total_updates)
                if total_updates > 0
                else 0.0
            )
            disconfirmatory_nongreedy_positive_ratio = (
                (episode_disconfirmatory_nongreedy_positive_updates / total_updates)
                if total_updates > 0
                else 0.0
            )
            td_error_metrics_log["confirmatory_update_ratio_per_episode"].append(
                float(confirmatory_ratio)
            )
            td_error_metrics_log[
                "confirmatory_greedy_positive_ratio_per_episode"
            ].append(float(confirmatory_greedy_positive_ratio))
            td_error_metrics_log[
                "confirmatory_nongreedy_negative_ratio_per_episode"
            ].append(float(confirmatory_nongreedy_negative_ratio))
            td_error_metrics_log[
                "disconfirmatory_greedy_negative_ratio_per_episode"
            ].append(float(disconfirmatory_greedy_negative_ratio))
            td_error_metrics_log[
                "disconfirmatory_nongreedy_positive_ratio_per_episode"
            ].append(float(disconfirmatory_nongreedy_positive_ratio))

        episode_metrics: Dict[str, Dict[str, List[float]]] = {
            "reward": reward_metrics_log,
            "td_error": td_error_metrics_log,
        }
        return episode_metrics

    def act(self, state: int) -> int:
        if self.q_table is None:
            raise RuntimeError("Q-table is not initialized. Call train() first.")
        return int(np.argmax(self.q_table[state]))
