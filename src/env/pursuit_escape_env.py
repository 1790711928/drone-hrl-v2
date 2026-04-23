from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, Tuple

from src.env.dynamics import (
    Agent3DState,
    Env3DState,
    relative_distance,
    rule_based_pursuer_control,
    step_kinematics,
)
from src.env.reward import compute_reward
from src.env.scenarios import SCENARIOS
from src.env.termination import EpisodeOutcome, TerminationConfig, TerminationState, evaluate_termination


@dataclass
class EnvConfig:
    dt: float = 0.1
    pursuer_speed_ratio: float = 1.2
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
        self.current_scenario: str = "rear_close_threat"

    def reset(
        self,
        scenario: str = "rear_close_threat",
        randomize: bool = False,
        rng: random.Random | None = None,
    ) -> Dict[str, float]:
        spec = SCENARIOS[scenario]
        self.current_scenario = scenario
        if randomize:
            rnd = rng or random
            evader = Agent3DState(
                x=spec.evader.x + rnd.uniform(-3.0, 3.0),
                y=spec.evader.y + rnd.uniform(-3.0, 3.0),
                z=spec.evader.z + rnd.uniform(-1.5, 1.5),
                speed=max(self.env_cfg.evader_speed_min, min(self.env_cfg.evader_speed_max, spec.evader.speed + rnd.uniform(-1.0, 1.0))),
                yaw=spec.evader.yaw + rnd.uniform(-0.15, 0.15),
                pitch=spec.evader.pitch + rnd.uniform(-0.08, 0.08),
            )
            pursuer = Agent3DState(
                x=spec.pursuer.x + rnd.uniform(-3.0, 3.0),
                y=spec.pursuer.y + rnd.uniform(-3.0, 3.0),
                z=spec.pursuer.z + rnd.uniform(-1.5, 1.5),
                speed=max(self.env_cfg.pursuer_speed_min, min(self.env_cfg.pursuer_speed_max, spec.pursuer.speed + rnd.uniform(-1.0, 1.0))),
                yaw=spec.pursuer.yaw + rnd.uniform(-0.15, 0.15),
                pitch=spec.pursuer.pitch + rnd.uniform(-0.08, 0.08),
            )
        else:
            evader = spec.evader
            pursuer = spec.pursuer

        self.state = Env3DState(evader=evader, pursuer=pursuer, step_count=0)
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

        los_escape_ok = self._los_escape_ok()
        outcome = evaluate_termination(
            distance=cur_distance,
            closing_speed=closing_speed,
            los_escape_ok=los_escape_ok,
            evader_position=(evader_next.x, evader_next.y, evader_next.z),
            step_count=self.state.step_count,
            cfg=self.term_cfg,
            tstate=self.tstate,
        )

        reward = compute_reward(
            scenario=self.current_scenario,
            prev_distance=prev_distance,
            cur_distance=cur_distance,
            action=(ev_accel, ev_yaw_rate, ev_pitch_rate),
            evader_position=(evader_next.x, evader_next.y, evader_next.z),
            bounds=(
                self.term_cfg.x_min,
                self.term_cfg.x_max,
                self.term_cfg.y_min,
                self.term_cfg.y_max,
                self.term_cfg.z_min,
                self.term_cfg.z_max,
            ),
            outcome=outcome,
        )
        done = outcome != EpisodeOutcome.RUNNING
        info = {
            "outcome": outcome.value,
            "distance": cur_distance,
            "closing_speed": closing_speed,
            "capture_streak": self.tstate.capture_streak,
            "escape_streak": self.tstate.escape_streak,
        }
        return self._observation(closing_speed=closing_speed), reward, done, info

    def _los_escape_ok(self) -> bool:
        if self.state is None:
            return False
        ev = self.state.evader
        pu = self.state.pursuer
        dx = ev.x - pu.x
        dy = ev.y - pu.y
        dz = ev.z - pu.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 1e-6:
            return False
        los_x, los_y, los_z = dx / norm, dy / norm, dz / norm
        heading_x = math.cos(ev.pitch) * math.cos(ev.yaw)
        heading_y = math.cos(ev.pitch) * math.sin(ev.yaw)
        heading_z = math.sin(ev.pitch)
        los_cos = heading_x * los_x + heading_y * los_y + heading_z * los_z
        return los_cos > 0.35

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
