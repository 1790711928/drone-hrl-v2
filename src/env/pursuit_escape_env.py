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
            pursuer_position=(pursuer_next.x, pursuer_next.y, pursuer_next.z),
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
        return self._compute_los_cos() > 0.35

    def _observation(self, closing_speed: float) -> Dict[str, float]:
        if self.state is None:
            raise RuntimeError("state is None")
        ev = self.state.evader
        pu = self.state.pursuer

        dx = ev.x - pu.x
        dy = ev.y - pu.y
        dz = ev.z - pu.z
        distance = relative_distance(self.state)
        los_cos = self._compute_los_cos()

        margin_x = min(ev.x - self.term_cfg.x_min, self.term_cfg.x_max - ev.x)
        margin_y = min(ev.y - self.term_cfg.y_min, self.term_cfg.y_max - ev.y)
        margin_z = min(ev.z - self.term_cfg.z_min, self.term_cfg.z_max - ev.z)
        min_boundary_margin = min(margin_x, margin_y, margin_z)
        normalized_step = self.state.step_count / max(float(self.term_cfg.max_steps), 1.0)
        world_xy_span = max(self.term_cfg.x_max - self.term_cfg.x_min, 1.0)
        world_z_span = max(self.term_cfg.z_max - self.term_cfg.z_min, 1.0)
        distance_scale = math.sqrt(world_xy_span * world_xy_span * 2.0 + world_z_span * world_z_span)
        speed_scale = max(self.env_cfg.evader_speed_max, self.env_cfg.pursuer_speed_max, 1.0)
        closing_speed_scale = max(speed_scale, 1.0)
        pitch_limit = math.pi / 2.0
        # Position-in-bounds direction signal: [-1, 1], keeps side information of boundary proximity.
        evader_x_norm = 2.0 * (ev.x - self.term_cfg.x_min) / world_xy_span - 1.0
        evader_y_norm = 2.0 * (ev.y - self.term_cfg.y_min) / world_xy_span - 1.0
        evader_z_norm = 2.0 * (ev.z - self.term_cfg.z_min) / world_z_span - 1.0

        # Threat direction in evader-local frame.
        rel_ex = pu.x - ev.x
        rel_ey = pu.y - ev.y
        rel_ez = pu.z - ev.z
        forward = (
            math.cos(ev.pitch) * math.cos(ev.yaw),
            math.cos(ev.pitch) * math.sin(ev.yaw),
            math.sin(ev.pitch),
        )
        right = (-math.sin(ev.yaw), math.cos(ev.yaw), 0.0)
        up = (
            -math.sin(ev.pitch) * math.cos(ev.yaw),
            -math.sin(ev.pitch) * math.sin(ev.yaw),
            math.cos(ev.pitch),
        )
        rel_norm = max(math.sqrt(rel_ex * rel_ex + rel_ey * rel_ey + rel_ez * rel_ez), 1e-6)
        threat_forward = (rel_ex * forward[0] + rel_ey * forward[1] + rel_ez * forward[2]) / rel_norm
        threat_right = (rel_ex * right[0] + rel_ey * right[1] + rel_ez * right[2]) / rel_norm
        threat_up = (rel_ex * up[0] + rel_ey * up[1] + rel_ez * up[2]) / rel_norm
        return {
            "dx": dx / world_xy_span,
            "dy": dy / world_xy_span,
            "dz": dz / world_z_span,
            "distance": distance / distance_scale,
            "closing_speed": closing_speed / closing_speed_scale,
            "evader_speed": ev.speed / speed_scale,
            "pursuer_speed": pu.speed / speed_scale,
            "evader_yaw_sin": math.sin(ev.yaw),
            "evader_yaw_cos": math.cos(ev.yaw),
            "pursuer_yaw_sin": math.sin(pu.yaw),
            "pursuer_yaw_cos": math.cos(pu.yaw),
            "evader_pitch": max(-1.0, min(1.0, ev.pitch / pitch_limit)),
            "pursuer_pitch": max(-1.0, min(1.0, pu.pitch / pitch_limit)),
            "los_cos": los_cos,
            "boundary_margin_x": margin_x / (world_xy_span * 0.5),
            "boundary_margin_y": margin_y / (world_xy_span * 0.5),
            "boundary_margin_z": margin_z / world_z_span,
            "min_boundary_margin": min_boundary_margin / max(world_xy_span * 0.5, world_z_span),
            "normalized_step": normalized_step,
            "evader_x_norm": evader_x_norm,
            "evader_y_norm": evader_y_norm,
            "evader_z_norm": evader_z_norm,
            "threat_forward": threat_forward,
            "threat_right": threat_right,
            "threat_up": threat_up,
        }

    def _compute_los_cos(self) -> float:
        if self.state is None:
            return 0.0
        ev = self.state.evader
        pu = self.state.pursuer
        dx = ev.x - pu.x
        dy = ev.y - pu.y
        dz = ev.z - pu.z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 1e-6:
            return 0.0
        los_x, los_y, los_z = dx / norm, dy / norm, dz / norm
        heading_x = math.cos(ev.pitch) * math.cos(ev.yaw)
        heading_y = math.cos(ev.pitch) * math.sin(ev.yaw)
        heading_z = math.sin(ev.pitch)
        return heading_x * los_x + heading_y * los_y + heading_z * los_z
