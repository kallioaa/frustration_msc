"""CliffWalking environment factory helpers."""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym


@dataclass
class CliffWalkingConfig:
    """Configuration for CliffWalking environment."""

    seed: Optional[int] = None
    is_slippery: bool = False


def get_cliffwalking_env(config: CliffWalkingConfig | dict) -> gym.Env:
    """Create a CliffWalking environment from a config object or dict."""
    if isinstance(config, dict):
        config = CliffWalkingConfig(**config)
    env = gym.make("CliffWalking-v1", is_slippery=config.is_slippery)
    if config.seed is not None:
        env.reset(seed=config.seed)
    return env
