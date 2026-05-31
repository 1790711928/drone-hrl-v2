from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.dynamics import Agent3DState, Env3DState
from src.env.scenarios import SCENARIOS
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
        Agent3DState(x=42.5, y=0.5, z=9.0, speed=9.6, yaw=3.02, pitch=0.0),
        Agent3DState(x=45.0, y=4.8, z=8.6, speed=11.2, yaw=2.75, pitch=0.0),
    ),
}


@dataclass(frozen=True)
class SequentialScenario:
    evader: Agent3DState
    phases: tuple[str, ...]


SEQUENTIAL_SCENARIOS: dict[str, SequentialScenario] = {
    "sequential_rear_to_boundary": SequentialScenario(
        evader=Agent3DState(x=0.0, y=0.0, z=12.0, speed=10.0, yaw=0.0, pitch=0.0),
        phases=("rear", "boundary"),
    ),
    "sequential_boundary_to_flank": SequentialScenario(
        evader=Agent3DState(x=42.0, y=0.0, z=10.0, speed=9.5, yaw=3.0, pitch=0.0),
        phases=("boundary", "flank"),
    ),
    "sequential_vertical_to_boundary": SequentialScenario(
        evader=Agent3DState(x=0.0, y=0.0, z=16.0, speed=10.0, yaw=0.0, pitch=0.08),
        phases=("vertical", "boundary"),
    ),
    "sequential_rear_to_flank_to_boundary": SequentialScenario(
        evader=Agent3DState(x=0.0, y=0.0, z=11.0, speed=9.8, yaw=0.0, pitch=0.0),
        phases=("rear", "flank", "boundary"),
    ),
    "sequential_rear_vertical_to_boundary": SequentialScenario(
        evader=Agent3DState(x=0.0, y=0.0, z=16.0, speed=10.0, yaw=0.0, pitch=0.08),
        phases=("rear_vertical", "boundary"),
    ),
}


PHASE_CANONICAL_SCENARIOS = {
    "rear": "rear_close_threat",
    "flank": "flank_threat",
    "boundary": "boundary_constrained",
    "vertical": "vertical_z_threat",
}


def _body_axes(agent: Agent3DState) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    forward = (math.cos(agent.pitch) * math.cos(agent.yaw), math.cos(agent.pitch) * math.sin(agent.yaw), math.sin(agent.pitch))
    right = (-math.sin(agent.yaw), math.cos(agent.yaw), 0.0)
    up = (-math.sin(agent.pitch) * math.cos(agent.yaw), -math.sin(agent.pitch) * math.sin(agent.yaw), math.cos(agent.pitch))
    return forward, right, up


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _canonical_local_offset(phase_name: str) -> tuple[float, float, float]:
    scenario = SCENARIOS[PHASE_CANONICAL_SCENARIOS[phase_name]]
    rel = (
        scenario.pursuer.x - scenario.evader.x,
        scenario.pursuer.y - scenario.evader.y,
        scenario.pursuer.z - scenario.evader.z,
    )
    return tuple(_dot(rel, axis) for axis in _body_axes(scenario.evader))


def inject_sequential_phase(evader: Agent3DState, phase_name: str) -> Env3DState:
    """Inject canonical low-level threat geometry while preserving sequential state where possible."""
    ev = evader
    if phase_name == "boundary":
        # Boundary recovery must match the canonical pi3 training distribution.
        canonical_evader = SCENARIOS[PHASE_CANONICAL_SCENARIOS[phase_name]].evader
        ev = replace(evader, x=canonical_evader.x, z=canonical_evader.z, yaw=canonical_evader.yaw, pitch=canonical_evader.pitch)

    if phase_name in PHASE_CANONICAL_SCENARIOS:
        scenario = SCENARIOS[PHASE_CANONICAL_SCENARIOS[phase_name]]
        local_f, local_r, local_u = _canonical_local_offset(phase_name)
        speed_delta = scenario.pursuer.speed - scenario.evader.speed
        yaw_delta = scenario.pursuer.yaw - scenario.evader.yaw
        pitch_delta = scenario.pursuer.pitch - scenario.evader.pitch
    else:
        # rear_vertical intentionally remains composite-only: it has no single
        # canonical low-level home scenario.
        local_f, local_r, local_u = (-4.5, 0.0, -7.5)
        speed_delta, yaw_delta, pitch_delta = 1.5, 0.0, 0.0

    forward, right, up = _body_axes(ev)
    pu = Agent3DState(
        x=ev.x + local_f * forward[0] + local_r * right[0] + local_u * up[0],
        y=ev.y + local_f * forward[1] + local_r * right[1] + local_u * up[1],
        z=ev.z + local_f * forward[2] + local_r * right[2] + local_u * up[2],
        speed=ev.speed + speed_delta,
        yaw=ev.yaw + yaw_delta,
        pitch=ev.pitch + pitch_delta,
    )
    return Env3DState(evader=ev, pursuer=pu, step_count=0)


PHASE_SUCCESS_THRESHOLDS: dict[str, dict[str, float]] = {
    "rear": {"distance_gain": 0.025, "closing_speed_max": 0.08},
    "flank": {"threat_right_reduction": 0.20, "threat_right_safe": 0.40, "distance_loss_max": 0.02},
    "boundary": {"margin_gain": 0.07, "controllable_margin": 0.25},
    "vertical": {"separation_low": 0.12, "separation_high": 0.30, "threat_up_safe": 0.75, "z_margin_safe": 0.20},
    "rear_vertical": {"distance_gain": 0.020, "closing_speed_max": 0.08, "separation_low": 0.12, "separation_high": 0.30, "threat_up_safe": 0.75, "z_margin_safe": 0.20},
}


class HighLevelOptionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}
    phase_completion_bonus = 5.0
    final_sequence_bonus = 20.0
    phase_success_streak_required = 3
    max_phase_lowlevel_steps = 80

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
        self.phase_index = 0
        self.completed_phases = 0
        self.phase_names: tuple[str, ...] = ()
        self.phase_start_metrics: dict[str, float] = {}
        self.phase_success_streak = 0
        self.phase_lowlevel_steps = 0
        self.phase_failed = False
        self.phase_success_by_type: dict[str, int] = {}
        self.phase_failure_by_type: dict[str, int] = {}

    def _sample_basic(self) -> str:
        return str(self.np_random.choice(BASIC_SCENARIOS))

    def _sample_mixed(self) -> tuple[str, str]:
        mixed_name = str(self.np_random.choice(list(MIXED_SCENARIOS.keys())))
        mix = MIXED_SCENARIOS[mixed_name]
        scenarios = list(mix.keys())
        probs = np.array(list(mix.values()), dtype=np.float64)
        return mixed_name, str(self.np_random.choice(scenarios, p=probs / probs.sum()))

    def _set_inner_state(self, state: Env3DState, scenario: str = "rear_close_threat") -> np.ndarray:
        self.inner.inner.state = state
        self.inner.inner.tstate = TerminationState()
        self.inner.inner.current_scenario = scenario
        return self.inner._flatten_obs(self.inner.inner._observation(closing_speed=0.0))

    def _reset_composite_state(self, scenario_name: str) -> np.ndarray:
        ev, pu = COMPOSITE_SCENARIOS[scenario_name]
        return self._set_inner_state(Env3DState(evader=ev, pursuer=pu, step_count=0), "boundary_constrained")

    def _inject_current_phase(self, evader: Agent3DState | None = None) -> np.ndarray:
        if evader is None:
            evader = SEQUENTIAL_SCENARIOS[self.current_scenario_name].evader
        state = inject_sequential_phase(evader, self.phase_names[self.phase_index])
        obs = self._set_inner_state(state)
        self._reset_phase_tracking()
        return obs

    def _reset_phase_tracking(self) -> None:
        state = self.inner.inner.state
        assert state is not None
        obs = self.inner.inner._observation(closing_speed=0.0)
        self.phase_start_metrics = {
            "distance": obs["distance"],
            "threat_right_abs": abs(obs["threat_right"]),
            "vertical_separation": abs(obs["dz"]),
            "min_boundary_margin": obs["min_boundary_margin"],
        }
        self.phase_success_streak = 0
        self.phase_lowlevel_steps = 0
        self.phase_failed = False

    def _phase_condition(self, obs: dict[str, float]) -> bool:
        phase = self.phase_names[self.phase_index]
        threshold = PHASE_SUCCESS_THRESHOLDS[phase]
        distance_gain = obs["distance"] - self.phase_start_metrics["distance"]
        vertical_sep = abs(obs["dz"])
        vertical_ok = (
            threshold.get("separation_low", 0.0) <= vertical_sep <= threshold.get("separation_high", 1.0)
            and abs(obs["threat_up"]) <= threshold.get("threat_up_safe", 1.0)
            and obs["boundary_margin_z"] >= threshold.get("z_margin_safe", 0.0)
        )
        if phase == "rear":
            return distance_gain >= threshold["distance_gain"] and obs["closing_speed"] <= threshold["closing_speed_max"]
        if phase == "flank":
            reduction = self.phase_start_metrics["threat_right_abs"] - abs(obs["threat_right"])
            return (reduction >= threshold["threat_right_reduction"] or abs(obs["threat_right"]) <= threshold["threat_right_safe"]) and distance_gain >= -threshold["distance_loss_max"]
        if phase == "boundary":
            margin_gain = obs["min_boundary_margin"] - self.phase_start_metrics["min_boundary_margin"]
            return obs["min_boundary_margin"] >= threshold["controllable_margin"] and margin_gain >= threshold["margin_gain"]
        if phase == "vertical":
            return vertical_ok
        return distance_gain >= threshold["distance_gain"] and obs["closing_speed"] <= threshold["closing_speed_max"] and vertical_ok

    def _increment_phase_stat(self, stats: dict[str, int], phase: str) -> None:
        stats[phase] = stats.get(phase, 0) + 1

    def _phase_info(self) -> dict[str, Any]:
        total = len(self.phase_names)
        return {
            "phase_name": self.phase_names[self.phase_index] if self.phase_names and self.phase_index < total else "completed",
            "phase_index": self.phase_index,
            "completed_phases": self.completed_phases,
            "total_phases": total,
            "phase_completion_rate": self.completed_phases / max(total, 1),
            "phase_success": False,
            "phase_failed": self.phase_failed,
            "phase_success_streak": self.phase_success_streak,
            "phase_lowlevel_steps": self.phase_lowlevel_steps,
            "phase_success_by_phase_type": dict(self.phase_success_by_type),
            "phase_failure_by_phase_type": dict(self.phase_failure_by_type),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.prev_option = None
        self.highlevel_step_count = 0
        self.switch_count = 0
        self.phase_index = 0
        self.completed_phases = 0
        self.phase_names = ()
        self.phase_start_metrics = {}
        self.phase_success_streak = 0
        self.phase_lowlevel_steps = 0
        self.phase_failed = False
        self.phase_success_by_type = {}
        self.phase_failure_by_type = {}

        if options and "scenario_set" in options:
            self.scenario_set = str(options["scenario_set"])

        requested_name = str(options["scenario_name"]) if options and "scenario_name" in options else None
        if self.scenario_set == "basic":
            base = requested_name or self._sample_basic()
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
        if self.scenario_set == "composite":
            comp = requested_name or str(self.np_random.choice(list(COMPOSITE_SCENARIOS.keys())))
            self.current_scenario_name = comp
            return self._reset_composite_state(comp), {"scenario_name": comp, "scenario_set": "composite"}

        seq = requested_name or str(self.np_random.choice(list(SEQUENTIAL_SCENARIOS.keys())))
        self.current_scenario_name = seq
        self.phase_names = SEQUENTIAL_SCENARIOS[seq].phases
        obs = self._inject_current_phase()
        return obs, {"scenario_name": seq, "scenario_set": "sequential", **self._phase_info()}

    def step(self, action: int):
        idx = int(np.clip(action, 0, 3))
        total_reward = 0.0
        duration_used = 0
        terminated = False
        truncated = False
        phase_transition = False
        info: dict[str, Any] = {"outcome": "running"}

        obs = self.inner._flatten_obs(self.inner.inner._observation(0.0))
        for _ in range(self.option_duration):
            low_action, _ = self.low_models[idx].predict(obs, deterministic=True)
            obs, reward, low_terminated, low_truncated, info = self.inner.step(low_action)
            total_reward += float(reward)
            duration_used += 1

            if self.scenario_set != "sequential":
                terminated, truncated = low_terminated, low_truncated
                if terminated or truncated:
                    break
                continue

            outcome = str(info.get("outcome", "running"))
            if outcome in {"captured", "out_of_bounds"}:
                terminated = True
                self.phase_failed = True
                self._increment_phase_stat(self.phase_failure_by_type, self.phase_names[self.phase_index])
                break
            if low_truncated:
                truncated = True
                self.phase_failed = True
                self._increment_phase_stat(self.phase_failure_by_type, self.phase_names[self.phase_index])
                break
            if outcome == "escaped":
                # Base escape does not solve a sequential phase. Remove the low-level
                # terminal bonus and keep evaluating the phase-specific objective.
                total_reward -= 50.0
                self.inner.inner.tstate = TerminationState()

            self.phase_lowlevel_steps += 1
            obs_dict = self.inner.inner._observation(float(info.get("closing_speed", 0.0)))
            if self._phase_condition(obs_dict):
                self.phase_success_streak += 1
            else:
                self.phase_success_streak = 0

            if self.phase_success_streak >= self.phase_success_streak_required:
                phase = self.phase_names[self.phase_index]
                self._increment_phase_stat(self.phase_success_by_type, phase)
                total_reward += self.phase_completion_bonus
                self.completed_phases += 1
                self.phase_index += 1
                phase_transition = True
                if self.phase_index >= len(self.phase_names):
                    total_reward += self.final_sequence_bonus
                    terminated = True
                    info = dict(info)
                    info["outcome"] = "escaped"
                else:
                    state = self.inner.inner.state
                    assert state is not None
                    obs = self._inject_current_phase(state.evader)
                    info = {"outcome": "running"}
                break

            if self.phase_lowlevel_steps >= self.max_phase_lowlevel_steps:
                truncated = True
                self.phase_failed = True
                self._increment_phase_stat(self.phase_failure_by_type, self.phase_names[self.phase_index])
                info = dict(info)
                info["outcome"] = "timeout"
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
                "phase_transition": phase_transition,
                **self._phase_info(),
                "phase_success": phase_transition,
            }
        )
        return obs, float(total_reward), bool(terminated), bool(truncated), info


HighLevelSwitchEnv = HighLevelOptionEnv
