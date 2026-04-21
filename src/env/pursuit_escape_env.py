from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.env.dynamics import (
    Agent3DState,
    Env3DState,
    relative_distance,
    rule_based_pursuer_control,
    step_kinematics,
)
from src.env.scenarios import SCENARIOS
from src.env.termination import EpisodeOutcome, TerminationConfig, TerminationState, evaluate_termination


@dataclass
class EnvConfig:
    dt: float = 0.1
    pursuer_speed_ratio: float = 1.1
    evader_speed_min: float = 2.0
    evader_speed_max: float = 20.0
    pursuer_speed_min: float = 2.0
    pursuer_speed_max: float = 25.0
    yaw_rate_max: float = 1.0
    pitch_rate_max: float = 0.7


class PursuitEscapeEnv:
    """Minimal 3D pursuit-escape environment for beginner-friendly HRL wiring."""

    def __init__(self, env_cfg: EnvConfig | None = None, term_cfg: TerminationConfig | None = None) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.term_cfg = term_cfg or TerminationConfig()
        self.state: Env3DState | None = None
        self.tstate: TerminationState = TerminationState()

    def reset(self, scenario: str = "s1_close_threat") -> Dict[str, float]:
        spec = SCENARIOS[scenario]
        self.state = Env3DState(evader=spec.evader, pursuer=spec.pursuer, step_count=0)
        self.tstate = TerminationState()
        return self._observation(closing_speed=0.0)

    def step(self, evader_action: Tuple[float, float, float]) -> tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        prev_distance = relative_distance(self.state)
        ev_accel, ev_yaw_rate, ev_pitch_rate = evader_action

        evader_next = step_kinematics(
            self.state.evader,
            ev_accel,
            ev_yaw_rate,
            ev_pitch_rate,
            self.env_cfg.dt,
            self.env_cfg.evader_speed_min,
            self.env_cfg.evader_speed_max,
            self.env_cfg.yaw_rate_max,
            self.env_cfg.pitch_rate_max,
        )

        pu_accel, pu_yaw_rate, pu_pitch_rate = rule_based_pursuer_control(
            evader_next,
            self.state.pursuer,
            self.env_cfg.pursuer_speed_ratio,
        )
        pursuer_next = step_kinematics(
            self.state.pursuer,
            pu_accel,
            pu_yaw_rate,
            pu_pitch_rate,
            self.env_cfg.dt,
            self.env_cfg.pursuer_speed_min,
            self.env_cfg.pursuer_speed_max,
            self.env_cfg.yaw_rate_max,
            self.env_cfg.pitch_rate_max,
        )

        self.state = Env3DState(evader=evader_next, pursuer=pursuer_next, step_count=self.state.step_count + 1)
        cur_distance = relative_distance(self.state)
        closing_speed = (prev_distance - cur_distance) / self.env_cfg.dt

        los_escape_ok = closing_speed <= 0.0
        outcome = evaluate_termination(
            distance=cur_distance,
            closing_speed=closing_speed,
            los_escape_ok=los_escape_ok,
            evader_position=(evader_next.x, evader_next.y, evader_next.z),
            step_count=self.state.step_count,
            cfg=self.term_cfg,
            tstate=self.tstate,
        )

        reward = (cur_distance - prev_distance) - 0.001 * (ev_accel**2 + ev_yaw_rate**2 + ev_pitch_rate**2)
        done = outcome != EpisodeOutcome.RUNNING
        info = {
            "outcome": outcome.value,
            "distance": cur_distance,
            "closing_speed": closing_speed,
            "capture_streak": self.tstate.capture_streak,
            "escape_streak": self.tstate.escape_streak,
        }
        return self._observation(closing_speed=closing_speed), reward, done, info

    def _observation(self, closing_speed: float) -> Dict[str, float]:
        if self.state is None:
            raise RuntimeError("state is None")
        return {
            "evader_x": self.state.evader.x,
            "evader_y": self.state.evader.y,
            "evader_z": self.state.evader.z,
            "pursuer_x": self.state.pursuer.x,
            "pursuer_y": self.state.pursuer.y,
            "pursuer_z": self.state.pursuer.z,
            "distance": relative_distance(self.state),
            "closing_speed": closing_speed,
        }
