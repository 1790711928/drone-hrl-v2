from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Agent3DState:
    x: float
    y: float
    z: float
    speed: float
    yaw: float
    pitch: float


@dataclass
class Env3DState:
    evader: Agent3DState
    pursuer: Agent3DState
    step_count: int = 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def step_kinematics(state: Agent3DState, accel: float, yaw_rate: float, pitch_rate: float, dt: float,
                    speed_min: float, speed_max: float, yaw_rate_max: float, pitch_rate_max: float) -> Agent3DState:
    speed = clamp(state.speed + accel * dt, speed_min, speed_max)
    yaw = state.yaw + clamp(yaw_rate, -yaw_rate_max, yaw_rate_max) * dt
    pitch = state.pitch + clamp(pitch_rate, -pitch_rate_max, pitch_rate_max) * dt
    pitch = clamp(pitch, -math.pi / 3, math.pi / 3)

    vx = speed * math.cos(pitch) * math.cos(yaw)
    vy = speed * math.cos(pitch) * math.sin(yaw)
    vz = speed * math.sin(pitch)

    return Agent3DState(
        x=state.x + vx * dt,
        y=state.y + vy * dt,
        z=state.z + vz * dt,
        speed=speed,
        yaw=yaw,
        pitch=pitch,
    )


def rule_based_pursuer_control(evader: Agent3DState, pursuer: Agent3DState, pursuer_speed_ratio: float) -> tuple[float, float, float]:
    dx, dy, dz = evader.x - pursuer.x, evader.y - pursuer.y, evader.z - pursuer.z
    target_yaw = math.atan2(dy, dx)
    horiz = math.hypot(dx, dy)
    target_pitch = math.atan2(dz, max(horiz, 1e-6))

    yaw_error = target_yaw - pursuer.yaw
    pitch_error = target_pitch - pursuer.pitch
    target_speed = max(evader.speed * pursuer_speed_ratio, pursuer.speed)
    accel = target_speed - pursuer.speed
    return accel, yaw_error, pitch_error


def relative_distance(state: Env3DState) -> float:
    dx = state.evader.x - state.pursuer.x
    dy = state.evader.y - state.pursuer.y
    dz = state.evader.z - state.pursuer.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)
