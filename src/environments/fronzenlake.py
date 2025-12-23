"""Environment factory helpers."""

import gymnasium as gym


def get_frozenlake_env(map_name: str = "4x4", is_slippery: bool = False) -> gym.Env:
    # Helper to create a FrozenLake environment with specified parameters.
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
    )
    return env
