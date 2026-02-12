"""Taxi-v3 environment factory helpers."""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym


@dataclass
class TaxiV3Config:
    """Configuration for Taxi-v3 environment."""

    seed: Optional[int] = None


def get_taxi_v3_env(config: TaxiV3Config | dict) -> gym.Env:
    """Create a Taxi-v3 environment from a config object or dict."""
    if isinstance(config, dict):
        config = TaxiV3Config(**config)
    env = gym.make("Taxi-v3")
    if config.seed is not None:
        env.reset(seed=config.seed)
    return env
