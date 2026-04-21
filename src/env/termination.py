from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EpisodeOutcome(str, Enum):
    RUNNING = "running"
    CAPTURED = "captured"
    ESCAPED = "escaped"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIMEOUT = "timeout"


@dataclass
class TerminationConfig:
    d_capture: float = 2.0
    d_safe: float = 20.0
    k_capture: int = 3
    k_escape: int = 5
    max_steps: int = 1000
    x_min: float = -100.0
    x_max: float = 100.0
    y_min: float = -100.0
    y_max: float = 100.0
    z_min: float = 0.0
    z_max: float = 50.0


@dataclass
class TerminationState:
    capture_streak: int = 0
    escape_streak: int = 0


def in_bounds(x: float, y: float, z: float, cfg: TerminationConfig) -> bool:
    return cfg.x_min <= x <= cfg.x_max and cfg.y_min <= y <= cfg.y_max and cfg.z_min <= z <= cfg.z_max


def evaluate_termination(
    *,
    distance: float,
    closing_speed: float,
    los_escape_ok: bool,
    evader_position: tuple[float, float, float],
    step_count: int,
    cfg: TerminationConfig,
    tstate: TerminationState,
) -> EpisodeOutcome:
    x, y, z = evader_position
    if not in_bounds(x, y, z, cfg):
        return EpisodeOutcome.OUT_OF_BOUNDS

    if distance < cfg.d_capture:
        tstate.capture_streak += 1
    else:
        tstate.capture_streak = 0

    escaped_now = distance > cfg.d_safe and closing_speed <= 0.0 and los_escape_ok
    if escaped_now:
        tstate.escape_streak += 1
    else:
        tstate.escape_streak = 0

    if tstate.capture_streak >= cfg.k_capture:
        return EpisodeOutcome.CAPTURED
    if tstate.escape_streak >= cfg.k_escape:
        return EpisodeOutcome.ESCAPED
    if step_count >= cfg.max_steps:
        return EpisodeOutcome.TIMEOUT
    return EpisodeOutcome.RUNNING
