from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.pursuit_escape_env import PursuitEscapeEnv


class PursuitEscapeGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium adapter for low-level SAC training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "s1_close_threat",
        scenario_weights: dict[str, float] | None = None,
        randomize_reset: bool = True,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.scenario_weights = scenario_weights
        self.randomize_reset = randomize_reset
        self.inner = PursuitEscapeEnv()

        # Action: [accel, yaw_rate, pitch_rate] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Obs: evader xyz + pursuer xyz + distance + closing_speed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            self.scenario = str(options["scenario"])
        chosen_scenario = self.scenario
        if self.scenario_weights:
            scenarios = list(self.scenario_weights.keys())
            probs = np.array(list(self.scenario_weights.values()), dtype=np.float64)
            probs = probs / probs.sum()
            chosen_scenario = str(self.np_random.choice(scenarios, p=probs))
        obs_dict = self.inner.reset(scenario=chosen_scenario, randomize=self.randomize_reset)
        return self._flatten_obs(obs_dict), {}

    def step(self, action: np.ndarray):
        accel = float(action[0]) * 1.0
        yaw_rate = float(action[1]) * self.inner.env_cfg.yaw_rate_max
        pitch_rate = float(action[2]) * self.inner.env_cfg.pitch_rate_max

        obs_dict, reward, done, info = self.inner.step((accel, yaw_rate, pitch_rate))
        obs = self._flatten_obs(obs_dict)

        terminated = bool(done and info.get("outcome") != "timeout")
        truncated = bool(done and info.get("outcome") == "timeout")
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _flatten_obs(obs_dict: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                obs_dict["evader_x"],
                obs_dict["evader_y"],
                obs_dict["evader_z"],
                obs_dict["pursuer_x"],
                obs_dict["pursuer_y"],
                obs_dict["pursuer_z"],
                obs_dict["distance"],
                obs_dict["closing_speed"],
            ],
            dtype=np.float32,
        )
