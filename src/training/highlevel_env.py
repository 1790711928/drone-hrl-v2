from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.training.sac_env import PursuitEscapeGymEnv

MIXED_SCENARIOS: dict[str, dict[str, float]] = {
    "mixed_rear_vertical": {
        "rear_close_threat": 0.55,
        "vertical_z_threat": 0.35,
        "boundary_constrained": 0.10,
    },
    "mixed_flank_boundary": {
        "flank_threat": 0.55,
        "boundary_constrained": 0.35,
        "rear_close_threat": 0.10,
    },
    "mixed_rear_boundary": {
        "rear_close_threat": 0.50,
        "boundary_constrained": 0.40,
        "vertical_z_threat": 0.10,
    },
    "mixed_vertical_boundary": {
        "vertical_z_threat": 0.50,
        "boundary_constrained": 0.40,
        "flank_threat": 0.10,
    },
    "mixed_rear_flank_boundary": {
        "rear_close_threat": 0.35,
        "flank_threat": 0.30,
        "boundary_constrained": 0.25,
        "vertical_z_threat": 0.10,
    },
}


class HighLevelOptionEnv(gym.Env[np.ndarray, int]):
    """High-level option env: one action triggers one low-level option rollout."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        low_models: list[Any],
        option_duration: int = 8,
        switch_penalty: float = 0.02,
        max_highlevel_steps: int = 80,
        mixed_scenarios: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.low_models = low_models
        self.option_duration = option_duration
        self.switch_penalty = switch_penalty
        self.max_highlevel_steps = max_highlevel_steps
        self.mixed_scenarios = mixed_scenarios or list(MIXED_SCENARIOS.keys())

        self.inner = PursuitEscapeGymEnv(scenario="rear_close_threat")
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.inner.observation_space

        self.prev_option = 0
        self.highlevel_step_count = 0
        self.switch_count = 0
        self.current_mixed_scenario = self.mixed_scenarios[0]

    def _choose_lowlevel_scenario(self) -> str:
        mix = MIXED_SCENARIOS[self.current_mixed_scenario]
        scenarios = list(mix.keys())
        probs = np.array(list(mix.values()), dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.np_random.choice(scenarios, p=probs))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "mixed_scenario" in options:
            self.current_mixed_scenario = str(options["mixed_scenario"])
        else:
            self.current_mixed_scenario = str(self.np_random.choice(self.mixed_scenarios))

        scenario = self._choose_lowlevel_scenario()
        obs, info = self.inner.reset(seed=seed, options={"scenario": scenario})
        self.prev_option = 0
        self.highlevel_step_count = 0
        self.switch_count = 0
        info.update({"mixed_scenario": self.current_mixed_scenario, "base_scenario": scenario})
        return obs, info

    def step(self, action: int):
        idx = int(np.clip(action, 0, 3))
        if idx != self.prev_option:
            self.switch_count += 1

        total_reward = 0.0
        duration_used = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"outcome": "running"}

        obs = self.inner._flatten_obs(self.inner.inner._observation(0.0))
        for _ in range(self.option_duration):
            low_action, _ = self.low_models[idx].predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.inner.step(low_action)
            total_reward += float(reward)
            duration_used += 1
            if terminated or truncated:
                break

        if idx != self.prev_option:
            total_reward -= self.switch_penalty
        self.prev_option = idx

        self.highlevel_step_count += 1
        if not (terminated or truncated) and self.highlevel_step_count >= self.max_highlevel_steps:
            truncated = True
            info = dict(info)
            info["outcome"] = "timeout"

        info = dict(info)
        info.update(
            {
                "selected_option": idx,
                "option_duration_used": duration_used,
                "switch_count": self.switch_count,
                "mixed_scenario": self.current_mixed_scenario,
            }
        )
        return obs, float(total_reward), bool(terminated), bool(truncated), info


# backward compatibility
HighLevelSwitchEnv = HighLevelOptionEnv
