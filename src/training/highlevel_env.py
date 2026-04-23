from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.training.sac_env import PursuitEscapeGymEnv


class HighLevelSwitchEnv(gym.Env[np.ndarray, int]):
    """High-level selector env: discrete action picks one of four frozen low-level policies."""

    def __init__(self, low_models: list[Any], switch_penalty: float = 0.02) -> None:
        super().__init__()
        self.low_models = low_models
        self.switch_penalty = switch_penalty
        self.scenarios = [
            "rear_close_threat",
            "flank_threat",
            "boundary_constrained",
            "vertical_z_threat",
        ]
        self.inner = PursuitEscapeGymEnv(scenario=self.scenarios[0])
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.inner.observation_space
        self.prev_idx = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        scenario = random.choice(self.scenarios)
        obs, info = self.inner.reset(seed=seed, options={"scenario": scenario})
        self.prev_idx = 0
        return obs, info

    def step(self, action: int):
        idx = int(np.clip(action, 0, 3))
        obs = self.inner._flatten_obs(self.inner.inner._observation(0.0))
        low_action, _ = self.low_models[idx].predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = self.inner.step(low_action)

        if idx != self.prev_idx:
            reward -= self.switch_penalty
        self.prev_idx = idx

        if info.get("outcome") == "escaped":
            reward += 30.0
        elif info.get("outcome") == "captured":
            reward -= 30.0

        return next_obs, float(reward), terminated, truncated, info
