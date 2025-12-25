"""Environment factory helpers."""

import gymnasium as gym
from dataclasses import dataclass


@dataclass
class FrozenLakeConfig:
    """Configuration for FrozenLake environment."""

    map_name: str = "4x4"
    is_slippery: bool = False


def get_frozenlake_env(config: FrozenLakeConfig) -> gym.Env:
    """Create a FrozenLake environment from a config object."""
    env = gym.make(
        "FrozenLake-v1",
        map_name=config.map_name,
        is_slippery=config.is_slippery,
    )
    return env
