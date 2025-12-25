"""CliffWalking environment factory helpers."""

from dataclasses import dataclass

import gymnasium as gym


@dataclass
class CliffWalkingConfig:
    """Configuration for CliffWalking environment."""


def get_cliffwalking_env(config: CliffWalkingConfig) -> gym.Env:
    """Create a CliffWalking environment from a config object."""
    env = gym.make("CliffWalking-v0")
    return env
