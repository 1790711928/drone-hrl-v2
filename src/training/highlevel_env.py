from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.dynamics import Agent3DState, Env3DState
from src.env.termination import TerminationState
from src.training.sac_env import PursuitEscapeGymEnv

BASIC_SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]

MIXED_SCENARIOS: dict[str, dict[str, float]] = {
    "mixed_rear_vertical": {"rear_close_threat": 0.55, "vertical_z_threat": 0.35, "boundary_constrained": 0.10},
    "mixed_flank_boundary": {"flank_threat": 0.55, "boundary_constrained": 0.35, "rear_close_threat": 0.10},
    "mixed_rear_boundary": {"rear_close_threat": 0.50, "boundary_constrained": 0.40, "vertical_z_threat": 0.10},
    "mixed_vertical_boundary": {"vertical_z_threat": 0.50, "boundary_constrained": 0.40, "flank_threat": 0.10},
    "mixed_rear_flank_boundary": {
        "rear_close_threat": 0.35,
        "flank_threat": 0.30,
        "boundary_constrained": 0.25,
        "vertical_z_threat": 0.10,
    },
}

COMPOSITE_SCENARIOS: dict[str, tuple[Agent3DState, Agent3DState]] = {
    "composite_rear_vertical": (
        Agent3DState(x=0.0, y=0.0, z=16.0, speed=10.0, yaw=0.0, pitch=0.10),
        Agent3DState(x=-4.0, y=0.5, z=5.5, speed=11.5, yaw=0.05, pitch=0.15),
    ),
    "composite_flank_boundary": (
        # Inward-facing evader with a pursuer on the outer flank: recoverable,
        # but still requires boundary recovery before resolving the flank.
        Agent3DState(x=42.0, y=1.5, z=9.5, speed=9.5, yaw=3.05, pitch=0.0),
        Agent3DState(x=44.0, y=8.5, z=9.0, speed=11.0, yaw=2.80, pitch=0.0),
    ),
    "composite_rear_boundary": (
        Agent3DState(x=45.0, y=-1.0, z=10.5, speed=9.8, yaw=0.55, pitch=0.0),
        Agent3DState(x=37.0, y=-1.5, z=10.5, speed=11.3, yaw=0.35, pitch=0.0),
    ),
    "composite_vertical_boundary": (
        Agent3DState(x=43.0, y=0.0, z=6.0, speed=9.7, yaw=0.25, pitch=0.08),
        Agent3DState(x=38.0, y=0.5, z=1.8, speed=11.0, yaw=0.20, pitch=0.12),
    ),
    "composite_rear_flank_boundary": (
        # Pursuer starts behind and to the flank in the evader-local frame,
        # while the evader has enough room for pi3 to recover from x_max.
        Agent3DState(x=42.5, y=0.5, z=9.0, speed=9.6, yaw=3.02, pitch=0.0),
        Agent3DState(x=45.0, y=4.8, z=8.6, speed=11.2, yaw=2.75, pitch=0.0),
    ),
}


class HighLevelOptionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        low_models: list[Any],
        option_duration: int = 8,
        switch_penalty: float = 0.02,
        max_highlevel_steps: int = 80,
        scenario_set: str = "mixed",
    ) -> None:
        super().__init__()
        self.low_models = low_models
        self.option_duration = option_duration
        self.switch_penalty = switch_penalty
        self.max_highlevel_steps = max_highlevel_steps
        self.scenario_set = scenario_set

        self.inner = PursuitEscapeGymEnv(scenario="rear_close_threat")
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.inner.observation_space

        self.prev_option: int | None = None
        self.highlevel_step_count = 0
        self.switch_count = 0
        self.current_scenario_name = "rear_close_threat"

    def _sample_basic(self) -> str:
        return str(self.np_random.choice(BASIC_SCENARIOS))

    def _sample_mixed(self) -> tuple[str, str]:
        mixed_name = str(self.np_random.choice(list(MIXED_SCENARIOS.keys())))
        mix = MIXED_SCENARIOS[mixed_name]
        scenarios = list(mix.keys())
        probs = np.array(list(mix.values()), dtype=np.float64)
        probs = probs / probs.sum()
        base = str(self.np_random.choice(scenarios, p=probs))
        return mixed_name, base

    def _sample_composite(self) -> str:
        return str(self.np_random.choice(list(COMPOSITE_SCENARIOS.keys())))

    def _reset_composite_state(self, scenario_name: str):
        ev, pu = COMPOSITE_SCENARIOS[scenario_name]
        self.inner.inner.state = Env3DState(evader=ev, pursuer=pu, step_count=0)
        self.inner.inner.tstate = TerminationState()
        self.inner.inner.current_scenario = "boundary_constrained"
        obs_dict = self.inner.inner._observation(closing_speed=0.0)
        return self.inner._flatten_obs(obs_dict)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.prev_option = None
        self.highlevel_step_count = 0
        self.switch_count = 0

        if options and "scenario_set" in options:
            self.scenario_set = str(options["scenario_set"])

        if self.scenario_set == "basic":
            base = self._sample_basic()
            obs, info = self.inner.reset(seed=seed, options={"scenario": base})
            self.current_scenario_name = base
            info.update({"scenario_name": base, "scenario_set": "basic"})
            return obs, info

        if self.scenario_set == "mixed":
            mixed_name, base = self._sample_mixed()
            obs, info = self.inner.reset(seed=seed, options={"scenario": base})
            self.current_scenario_name = mixed_name
            info.update({"scenario_name": mixed_name, "base_scenario": base, "scenario_set": "mixed"})
            return obs, info

        comp = self._sample_composite()
        obs = self._reset_composite_state(comp)
        self.current_scenario_name = comp
        return obs, {"scenario_name": comp, "scenario_set": "composite"}

    def step(self, action: int):
        idx = int(np.clip(action, 0, 3))

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

        if self.prev_option is not None and idx != self.prev_option:
            self.switch_count += 1
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
                "scenario_name": self.current_scenario_name,
                "scenario_set": self.scenario_set,
            }
        )
        return obs, float(total_reward), bool(terminated), bool(truncated), info


HighLevelSwitchEnv = HighLevelOptionEnv
