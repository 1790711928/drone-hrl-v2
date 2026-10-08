"""Paired, raw low-level skill behavior evaluation.

This module deliberately collects raw behavior measurements only.  It does not
define a composite alignment score and does not import the legacy ESAS code.
Each scenario/trial first samples one randomized initial state, then every
policy is evaluated from a deep copy of that exact state.

The environment defines ``closing_speed`` as
``(previous_distance - current_distance) / dt``.  Positive values mean that
distance is decreasing (the pursuer is closing); zero or negative values mean
that the episode is no longer closing at that step.  The raw
``non_closing_ratio`` therefore counts post-step closing-speed samples <= 0.
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import random
import statistics
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

from src.env.dynamics import Env3DState, relative_distance
from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.env.termination import TerminationState
from src.training.sac_env import PursuitEscapeGymEnv

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]
POLICIES = ["pi1", "pi2", "pi3", "pi4"]
MODEL_FILENAMES = [
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
]

# This default matches the existing normalized boundary-risk safety threshold
# in src/env/reward.py: a normalized minimum margin of 0.35 is outside the
# soft-risk region.  It remains a CLI parameter for later protocol changes.
DEFAULT_BOUNDARY_SAFE_THRESHOLD = 0.35
# This matches the existing vertical skill diagnostic's normalized z-margin
# safety threshold (VERTICAL_Z_MARGIN_SAFE).
DEFAULT_VERTICAL_SAFE_THRESHOLD = 0.20

BEHAVIOR_METRICS = [
    "distance_gain",
    "non_closing_ratio",
    "lateral_threat_reduction",
    "boundary_margin_gain",
    "safe_boundary_ratio",
    "vertical_separation_gain",
    "vertical_safe_ratio",
]
DIAGNOSTIC_METRICS = [
    "threat_forward_early_mean",
    "threat_forward_late_mean",
    "threat_forward_improvement",
    "rear_gap_early_mean",
    "rear_gap_late_mean",
    "rear_gap_gain",
    "rear_gap_non_decreasing_ratio",
    "rear_away_displacement_early_mean",
    "rear_away_displacement_late_mean",
    "rear_away_displacement_gain",
    "evader_path_length",
    "rear_away_net_displacement",
    "rear_escape_efficiency",
    "side_gap_early_mean",
    "side_gap_late_mean",
    "side_gap_gain",
    "side_gap_non_decreasing_ratio",
    "threat_up_early_mean",
    "threat_up_late_mean",
    "threat_up_improvement",
    "vertical_threat_early_mean",
    "vertical_threat_late_mean",
    "vertical_threat_reduction",
    "vertical_away_displacement_early_mean",
    "vertical_away_displacement_late_mean",
    "vertical_away_displacement_gain",
    "vertical_away_net_displacement",
    "final_evader_z",
    "final_pursuer_z",
]
REPORT_METRICS = [*BEHAVIOR_METRICS, *DIAGNOSTIC_METRICS]
VALID_METRIC_FIELDS = {
    "rear_away_displacement_gain": "rear_away_displacement_gain_valid",
    "rear_escape_efficiency": "rear_escape_efficiency_valid",
    "side_gap_gain": "side_gap_gain_valid",
    "side_gap_non_decreasing_ratio": "side_gap_non_decreasing_ratio_valid",
    "boundary_margin_gain": "boundary_margin_gain_valid",
    "safe_boundary_ratio": "safe_boundary_ratio_valid",
    "vertical_away_displacement_gain": "vertical_away_displacement_gain_valid",
    "distance_gain": "distance_gain_valid",
}
VALID_METRIC_FIELDS_BY_RAW = {field: raw for raw, field in VALID_METRIC_FIELDS.items()}
MATRIX_FILENAMES = {
    "distance_gain": "distance_gain_matrix.csv",
    "non_closing_ratio": "non_closing_ratio_matrix.csv",
    "lateral_threat_reduction": "lateral_threat_reduction_matrix.csv",
    "boundary_margin_gain": "boundary_margin_gain_matrix.csv",
    "safe_boundary_ratio": "safe_boundary_ratio_matrix.csv",
    "vertical_separation_gain": "vertical_separation_gain_matrix.csv",
    "vertical_safe_ratio": "vertical_safe_ratio_matrix.csv",
}
SHEET_MATRIX_METRICS = {
    "Rear": [
        "distance_gain",
        "non_closing_ratio",
        "threat_forward_early_mean",
        "threat_forward_late_mean",
        "threat_forward_improvement",
        "rear_gap_early_mean",
        "rear_gap_late_mean",
        "rear_gap_gain",
        "rear_gap_non_decreasing_ratio",
        "rear_away_displacement_early_mean",
        "rear_away_displacement_late_mean",
        "rear_away_displacement_gain",
        "evader_path_length",
        "rear_away_net_displacement",
        "rear_escape_efficiency",
        "rear_away_displacement_gain_valid",
        "rear_escape_efficiency_valid",
    ],
    "Flank": [
        "lateral_threat_reduction",
        "side_gap_early_mean",
        "side_gap_late_mean",
        "side_gap_gain",
        "side_gap_non_decreasing_ratio",
        "side_gap_gain_valid",
        "side_gap_non_decreasing_ratio_valid",
    ],
    "Boundary": [
        "boundary_margin_gain",
        "safe_boundary_ratio",
        "boundary_margin_gain_valid",
        "safe_boundary_ratio_valid",
    ],
    "Vertical": [
        "vertical_separation_gain",
        "vertical_safe_ratio",
        "threat_up_early_mean",
        "threat_up_late_mean",
        "threat_up_improvement",
        "vertical_threat_early_mean",
        "vertical_threat_late_mean",
        "vertical_threat_reduction",
        "vertical_away_displacement_early_mean",
        "vertical_away_displacement_late_mean",
        "vertical_away_displacement_gain",
        "vertical_away_net_displacement",
        "vertical_away_displacement_gain_valid",
        "distance_gain_valid",
    ],
}


def early_window(values: Sequence[float], fraction: float = 0.20) -> list[float]:
    """Return the first fraction of a trace, with at least one sample."""
    if not values:
        return []
    count = max(1, int(len(values) * fraction))
    return list(values[:count])


def late_window(values: Sequence[float], fraction: float = 0.20) -> list[float]:
    """Return the final fraction of a trace, with at least one sample."""
    if not values:
        return []
    start = max(0, int(len(values) * (1.0 - fraction)))
    return list(values[start:])


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def trace_change(values: Sequence[float]) -> dict[str, float]:
    """Summarize a signed diagnostic trace using the fixed early/late windows.

    ``improvement`` is intentionally a signed change (late mean minus early
    mean).  It is diagnostic only: no direction or weight is used in the seven
    core behavior metrics.
    """
    early = mean_or_zero(early_window(values))
    late = mean_or_zero(late_window(values))
    return {"early_mean": early, "late_mean": late, "improvement": late - early}


def initial_reference_axes(initial_state: Env3DState) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the fixed initial evader forward and right axes.

    The axes use the same body-frame convention as ``PursuitEscapeEnv``:
    forward is the evader heading and right is ``(-sin(yaw), cos(yaw), 0)``.
    They are computed once from the initial pose and never updated during an
    episode.
    """
    evader = initial_state.evader
    forward = (
        math.cos(evader.pitch) * math.cos(evader.yaw),
        math.cos(evader.pitch) * math.sin(evader.yaw),
        math.sin(evader.pitch),
    )
    right = (-math.sin(evader.yaw), math.cos(evader.yaw), 0.0)
    return forward, right


def fixed_reference_gaps(
    initial_state: Env3DState,
    state_trace: Sequence[Env3DState],
) -> tuple[list[float], list[float]]:
    """Compute rear and signed-side gaps against one fixed initial frame."""
    if not state_trace:
        return [], []
    forward, right = initial_reference_axes(initial_state)
    initial_evader = initial_state.evader
    initial_pursuer = initial_state.pursuer
    initial_rx = initial_pursuer.x - initial_evader.x
    initial_ry = initial_pursuer.y - initial_evader.y
    initial_rz = initial_pursuer.z - initial_evader.z
    initial_side_projection = initial_rx * right[0] + initial_ry * right[1] + initial_rz * right[2]
    if abs(initial_side_projection) <= 1e-9:
        sigma = 1.0
    else:
        sigma = 1.0 if initial_side_projection > 0.0 else -1.0

    rear_gaps: list[float] = []
    side_gaps: list[float] = []
    for state in state_trace:
        rx = state.pursuer.x - state.evader.x
        ry = state.pursuer.y - state.evader.y
        rz = state.pursuer.z - state.evader.z
        forward_projection = rx * forward[0] + ry * forward[1] + rz * forward[2]
        side_projection = rx * right[0] + ry * right[1] + rz * right[2]
        rear_gaps.append(-forward_projection)
        side_gaps.append(sigma * side_projection)
    return rear_gaps, side_gaps


def initial_away_direction(initial_state: Env3DState) -> tuple[float, float, float]:
    """Return the fixed unit vector from the initial pursuer to evader."""
    evader = initial_state.evader
    pursuer = initial_state.pursuer
    dx = evader.x - pursuer.x
    dy = evader.y - pursuer.y
    dz = evader.z - pursuer.z
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (dx / norm, dy / norm, dz / norm)


def rear_away_displacement_trace(
    initial_state: Env3DState,
    state_trace: Sequence[Env3DState],
) -> list[float]:
    """Project evader displacement onto the fixed initial threat-away axis."""
    axis = initial_away_direction(initial_state)
    initial = initial_state.evader
    return [
        (state.evader.x - initial.x) * axis[0]
        + (state.evader.y - initial.y) * axis[1]
        + (state.evader.z - initial.z) * axis[2]
        for state in state_trace
    ]


def evader_path_length(state_trace: Sequence[Env3DState]) -> float:
    """Return the physical length of the evader trajectory."""
    total = 0.0
    for previous, current in zip(state_trace, state_trace[1:]):
        dx = current.evader.x - previous.evader.x
        dy = current.evader.y - previous.evader.y
        dz = current.evader.z - previous.evader.z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def rear_escape_efficiency(
    initial_state: Env3DState,
    state_trace: Sequence[Env3DState],
    eps: float = 1e-9,
) -> tuple[float, float, float]:
    """Return path length, net away displacement, and their ratio."""
    away_trace = rear_away_displacement_trace(initial_state, state_trace)
    path_length = evader_path_length(state_trace)
    net_displacement = away_trace[-1] if away_trace else 0.0
    return path_length, net_displacement, net_displacement / (path_length + eps)


def vertical_away_displacement_trace(
    initial_state: Env3DState,
    state_trace: Sequence[Env3DState],
) -> list[float]:
    """Project evader z displacement away from the initial vertical threat."""
    delta_z0 = initial_state.pursuer.z - initial_state.evader.z
    if abs(delta_z0) <= 1e-9:
        sigma_z = 1.0
    else:
        sigma_z = 1.0 if delta_z0 > 0.0 else -1.0
    initial_z = initial_state.evader.z
    return [-sigma_z * (state.evader.z - initial_z) for state in state_trace]


def compute_task_valid(outcome: Any) -> int:
    """Return the shared outer task gate: only escaped episodes are valid."""
    value = getattr(outcome, "value", outcome)
    return int(str(value) == "escaped")


def gated_metric(task_valid: int, raw_metric: float) -> float:
    """Apply task validity without deleting failed episodes or rescaling."""
    return float(task_valid) * float(raw_metric)


def non_decreasing_ratio(values: Sequence[float]) -> float:
    """Return the fraction of adjacent samples that do not decrease."""
    if len(values) < 2:
        return 0.0
    return sum(current >= previous for previous, current in zip(values, values[1:])) / (len(values) - 1)


def absolute_trace_reduction(values: Sequence[float]) -> dict[str, float]:
    """Summarize absolute early/late threat magnitude and its reduction."""
    magnitudes = [abs(value) for value in values]
    early = mean_or_zero(early_window(magnitudes))
    late = mean_or_zero(late_window(magnitudes))
    return {"early_mean": early, "late_mean": late, "reduction": early - late}


def non_closing_ratio(closing_speeds: Sequence[float]) -> float:
    """Return the fraction of post-step samples with no positive closing.

    ``PursuitEscapeEnv.step`` computes closing speed as
    ``(previous_distance - current_distance) / dt``.  Thus ``<= 0`` means
    distance is maintained or increasing and is favorable for escape.
    """
    if not closing_speeds:
        return 0.0
    return sum(float(value) <= 0.0 for value in closing_speeds) / len(closing_speeds)


def compute_raw_behavior_metrics(
    *,
    distances: Sequence[float],
    closing_speeds: Sequence[float],
    threat_right: Sequence[float],
    min_boundary_margins: Sequence[float],
    boundary_margin_z: Sequence[float],
    vertical_separations: Sequence[float],
    boundary_safe_threshold: float,
    vertical_safe_threshold: float,
) -> dict[str, float]:
    """Compute raw episode metrics without any composite weighting.

    State traces include the initial state and every post-step state.  Ratios
    over state traces therefore include the initial and terminal samples; the
    non-closing ratio uses only post-step closing-speed samples because the
    initial observation has no measured distance change.
    """
    if not distances:
        raise ValueError("distances must contain at least the initial sample")
    if not threat_right or not min_boundary_margins or not boundary_margin_z or not vertical_separations:
        raise ValueError("behavior traces must contain at least one state sample")

    early_right = early_window([abs(value) for value in threat_right])
    late_right = late_window([abs(value) for value in threat_right])
    row = {
        "distance_gain": float(distances[-1] - distances[0]),
        "non_closing_ratio": float(non_closing_ratio(closing_speeds)),
        "lateral_threat_reduction": float(mean_or_zero(early_right) - mean_or_zero(late_right)),
        "boundary_margin_gain": float(min_boundary_margins[-1] - min_boundary_margins[0]),
        "safe_boundary_ratio": float(
            sum(float(value) >= boundary_safe_threshold for value in min_boundary_margins)
            / len(min_boundary_margins)
        ),
        # This is a physical z separation in the same coordinate units as the
        # environment state, not the normalized observation ``dz``.
        "vertical_separation_gain": float(vertical_separations[-1] - vertical_separations[0]),
        "vertical_safe_ratio": float(
            sum(float(value) >= vertical_safe_threshold for value in boundary_margin_z)
            / len(boundary_margin_z)
        ),
    }
    return row


def _state_observation(env: PursuitEscapeGymEnv, closing_speed: float) -> dict[str, float]:
    return env.inner._observation(closing_speed=closing_speed)


def sample_paired_initial_state(scenario: str, seed: int) -> Env3DState:
    """Sample one deterministic evaluation state using a local Python RNG.

    The production reset path still uses its existing behavior.  Evaluation
    passes this state explicitly to every policy, so Python ``random`` and Gym
    seeding cannot create policy-specific initial conditions.
    """
    reference = PursuitEscapeEnv()
    reference.reset(scenario=scenario, randomize=True, rng=random.Random(seed))
    if reference.state is None:
        raise RuntimeError("reference environment did not initialize a state")
    return copy.deepcopy(reference.state)


def _paired_env(scenario: str, initial_state: Env3DState) -> PursuitEscapeGymEnv:
    env = PursuitEscapeGymEnv(scenario=scenario, randomize_reset=False)
    env.inner.state = copy.deepcopy(initial_state)
    env.inner.tstate = TerminationState()
    env.inner.current_scenario = scenario
    return env


def detect_out_of_bounds_axis(state: Env3DState, env: PursuitEscapeGymEnv, outcome: str) -> str:
    """Return the terminal evader boundary axis using termination's bounds."""
    if outcome != "out_of_bounds":
        return "none"
    position = state.evader
    bounds = env.inner.term_cfg
    if position.x < bounds.x_min:
        return "x_min"
    if position.x > bounds.x_max:
        return "x_max"
    if position.y < bounds.y_min:
        return "y_min"
    if position.y > bounds.y_max:
        return "y_max"
    if position.z < bounds.z_min:
        return "z_min"
    if position.z > bounds.z_max:
        return "z_max"
    return "none"


def run_paired_episode(
    model: Any,
    *,
    scenario: str,
    policy: str,
    trial_id: int,
    seed: int,
    initial_state: Env3DState,
    boundary_safe_threshold: float,
    vertical_safe_threshold: float,
) -> dict[str, Any]:
    """Run one policy from a supplied paired initial state."""
    env = _paired_env(scenario, initial_state)
    state = env.inner.state
    assert state is not None
    state_trace = [copy.deepcopy(state)]
    obs_dict = _state_observation(env, 0.0)
    obs = env._flatten_obs(obs_dict)

    distances = [relative_distance(state)]
    threat_forward = [float(obs_dict["threat_forward"])]
    threat_right = [float(obs_dict["threat_right"])]
    threat_up = [float(obs_dict["threat_up"])]
    min_boundary_margins = [float(obs_dict["min_boundary_margin"])]
    boundary_margin_z = [float(obs_dict["boundary_margin_z"])]
    vertical_separations = [abs(state.evader.z - state.pursuer.z)]
    closing_speeds: list[float] = []
    total_reward = 0.0
    episode_length = 0
    info: dict[str, Any] = {"outcome": "timeout"}
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        episode_length += 1
        state = env.inner.state
        assert state is not None
        state_trace.append(copy.deepcopy(state))
        closing_speed = float(info.get("closing_speed", 0.0))
        obs_dict = _state_observation(env, closing_speed)
        distances.append(relative_distance(state))
        closing_speeds.append(closing_speed)
        threat_forward.append(float(obs_dict["threat_forward"]))
        threat_right.append(float(obs_dict["threat_right"]))
        threat_up.append(float(obs_dict["threat_up"]))
        min_boundary_margins.append(float(obs_dict["min_boundary_margin"]))
        boundary_margin_z.append(float(obs_dict["boundary_margin_z"]))
        vertical_separations.append(abs(state.evader.z - state.pursuer.z))

    outcome = str(info.get("outcome", "timeout"))
    metrics = compute_raw_behavior_metrics(
        distances=distances,
        closing_speeds=closing_speeds,
        threat_right=threat_right,
        min_boundary_margins=min_boundary_margins,
        boundary_margin_z=boundary_margin_z,
        vertical_separations=vertical_separations,
        boundary_safe_threshold=boundary_safe_threshold,
        vertical_safe_threshold=vertical_safe_threshold,
    )
    forward_change = trace_change(threat_forward)
    up_change = trace_change(threat_up)
    vertical_threat_change = absolute_trace_reduction(threat_up)
    rear_gaps, side_gaps = fixed_reference_gaps(initial_state, state_trace)
    rear_away_trace = rear_away_displacement_trace(initial_state, state_trace)
    path_length, rear_away_net, rear_efficiency = rear_escape_efficiency(initial_state, state_trace)
    vertical_away_trace = vertical_away_displacement_trace(initial_state, state_trace)
    rear_gap_early = mean_or_zero(early_window(rear_gaps))
    rear_gap_late = mean_or_zero(late_window(rear_gaps))
    side_gap_early = mean_or_zero(early_window(side_gaps))
    side_gap_late = mean_or_zero(late_window(side_gaps))
    rear_away_early = mean_or_zero(early_window(rear_away_trace))
    rear_away_late = mean_or_zero(late_window(rear_away_trace))
    vertical_away_early = mean_or_zero(early_window(vertical_away_trace))
    vertical_away_late = mean_or_zero(late_window(vertical_away_trace))
    out_of_bounds_axis = detect_out_of_bounds_axis(state, env, outcome)
    task_valid_value = compute_task_valid(outcome)
    initial_state_fields = {
        "initial_evader_x": initial_state.evader.x,
        "initial_evader_y": initial_state.evader.y,
        "initial_evader_z": initial_state.evader.z,
        "initial_evader_speed": initial_state.evader.speed,
        "initial_evader_yaw": initial_state.evader.yaw,
        "initial_evader_pitch": initial_state.evader.pitch,
        "initial_pursuer_x": initial_state.pursuer.x,
        "initial_pursuer_y": initial_state.pursuer.y,
        "initial_pursuer_z": initial_state.pursuer.z,
        "initial_pursuer_speed": initial_state.pursuer.speed,
        "initial_pursuer_yaw": initial_state.pursuer.yaw,
        "initial_pursuer_pitch": initial_state.pursuer.pitch,
    }
    return {
        "scenario": scenario,
        "policy": policy,
        "seed": seed,
        "trial_id": trial_id,
        "success": task_valid_value,
        "task_valid": task_valid_value,
        "captured": int(outcome == "captured"),
        "out_of_bounds": int(outcome == "out_of_bounds"),
        "timeout": int(outcome == "timeout"),
        "outcome": outcome,
        "episode_length": episode_length,
        "initial_distance": distances[0],
        "final_distance": distances[-1],
        "total_reward": total_reward,
        "initial_min_boundary_margin": min_boundary_margins[0],
        "final_min_boundary_margin": min_boundary_margins[-1],
        "initial_vertical_separation": vertical_separations[0],
        "final_vertical_separation": vertical_separations[-1],
        "final_evader_z": state.evader.z,
        "final_pursuer_z": state.pursuer.z,
        "out_of_bounds_axis": out_of_bounds_axis,
        "boundary_safe_threshold": boundary_safe_threshold,
        "vertical_safe_threshold": vertical_safe_threshold,
        "threat_forward_early_mean": forward_change["early_mean"],
        "threat_forward_late_mean": forward_change["late_mean"],
        "threat_forward_improvement": forward_change["improvement"],
        "rear_gap_early_mean": rear_gap_early,
        "rear_gap_late_mean": rear_gap_late,
        "rear_gap_gain": rear_gap_late - rear_gap_early,
        "rear_gap_non_decreasing_ratio": non_decreasing_ratio(rear_gaps),
        "rear_away_displacement_early_mean": rear_away_early,
        "rear_away_displacement_late_mean": rear_away_late,
        "rear_away_displacement_gain": rear_away_late - rear_away_early,
        "evader_path_length": path_length,
        "rear_away_net_displacement": rear_away_net,
        "rear_escape_efficiency": rear_efficiency,
        "side_gap_early_mean": side_gap_early,
        "side_gap_late_mean": side_gap_late,
        "side_gap_gain": side_gap_late - side_gap_early,
        "side_gap_non_decreasing_ratio": non_decreasing_ratio(side_gaps),
        "threat_up_early_mean": up_change["early_mean"],
        "threat_up_late_mean": up_change["late_mean"],
        "threat_up_improvement": up_change["improvement"],
        "vertical_threat_early_mean": vertical_threat_change["early_mean"],
        "vertical_threat_late_mean": vertical_threat_change["late_mean"],
        "vertical_threat_reduction": vertical_threat_change["reduction"],
        "vertical_away_displacement_early_mean": vertical_away_early,
        "vertical_away_displacement_late_mean": vertical_away_late,
        "vertical_away_displacement_gain": vertical_away_late - vertical_away_early,
        "vertical_away_net_displacement": vertical_away_trace[-1] if vertical_away_trace else 0.0,
        **initial_state_fields,
        **metrics,
    }
    for raw_metric, valid_field in VALID_METRIC_FIELDS.items():
        row[valid_field] = gated_metric(task_valid_value, row[raw_metric])
    return row


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.policy_checkpoints is not None:
        return [Path(value) for value in args.policy_checkpoints]
    checkpoint_dir = Path(args.checkpoint_dir)
    return [
        Path(args.policy1_checkpoint) if args.policy1_checkpoint else checkpoint_dir / MODEL_FILENAMES[0],
        Path(args.policy2_checkpoint) if args.policy2_checkpoint else checkpoint_dir / MODEL_FILENAMES[1],
        Path(args.policy3_checkpoint) if args.policy3_checkpoint else checkpoint_dir / MODEL_FILENAMES[2],
        Path(args.policy4_checkpoint) if args.policy4_checkpoint else checkpoint_dir / MODEL_FILENAMES[3],
    ]


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str] = BEHAVIOR_METRICS,
) -> list[dict[str, Any]]:
    """Return long-form statistics without combining metrics into a score."""
    summary: list[dict[str, Any]] = []
    outcome_fields = (
        ("success", "success_rate"),
        ("captured", "capture_rate"),
        ("out_of_bounds", "out_of_bounds_rate"),
        ("timeout", "timeout_rate"),
    )
    for scenario in SCENARIOS:
        for policy in POLICIES:
            group = [row for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            if not group:
                continue
            outcome_rates = {
                output_name: mean_or_zero(row[input_name] for row in group)
                for input_name, output_name in outcome_fields
            }
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                valid_field = VALID_METRIC_FIELDS.get(metric)
                valid_values = (
                    [float(row[valid_field]) for row in group]
                    if valid_field and all(valid_field in row for row in group)
                    else []
                )
                summary.append(
                    {
                        "scenario": scenario,
                        "policy": policy,
                        "metric": metric,
                        "mean": statistics.mean(values),
                        "valid_mean": statistics.mean(valid_values) if valid_values else "",
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "episode_count": len(values),
                        **outcome_rates,
                    }
                )
    return summary


def write_raw_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["scenario", "policy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = [
    "scenario", "policy", "metric", "mean", "valid_mean", "std", "median", "min", "max", "episode_count",
    "success_rate", "capture_rate", "out_of_bounds_rate", "timeout_rate",
]


def _summary_table_rows(summary_rows: Sequence[dict[str, Any]]) -> list[list[Any]]:
    return [
        SUMMARY_FIELDS,
        *[[row.get(field, "") for field in SUMMARY_FIELDS] for row in summary_rows],
    ]


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the existing long-form summary CSV, including diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _matrix_rows(summary_rows: Sequence[dict[str, Any]], metric: str) -> list[list[Any]]:
    rows: list[list[Any]] = [["scenario", *POLICIES]]
    source_metric = VALID_METRIC_FIELDS_BY_RAW.get(metric, metric)
    for scenario in SCENARIOS:
        row: list[Any] = [scenario]
        for policy in POLICIES:
            matches = [
                item
                for item in summary_rows
                if item["scenario"] == scenario and item["policy"] == policy and item["metric"] == source_metric
            ]
            if not matches:
                row.append("")
            elif metric in VALID_METRIC_FIELDS_BY_RAW:
                row.append(matches[0]["valid_mean"])
            else:
                row.append(matches[0]["mean"])
        rows.append(row)
    return rows


def _outcome_axis_rows(raw_rows: Sequence[dict[str, Any]]) -> list[list[Any]]:
    axes = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "none"]
    rows: list[list[Any]] = [["scenario", "policy", *axes]]
    for scenario in SCENARIOS:
        for policy in POLICIES:
            group = [row for row in raw_rows if row["scenario"] == scenario and row["policy"] == policy]
            counts = {axis: sum(row.get("out_of_bounds_axis") == axis for row in group) for axis in axes}
            rows.append([scenario, policy, *[counts[axis] for axis in axes]])
    return rows


def _xlsx_column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xlsx_value_cell(reference: str, value: Any) -> str:
    if value is None or value == "":
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return f'<c r="{reference}"><v>{number:.15g}</v></c>'
    text = escape(str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    widths: dict[int, int] = {}
    xml_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row, start=1):
            widths[column_index] = min(
                50,
                max(widths.get(column_index, 10), len(str(value)) + 2 if value not in (None, "") else 10),
            )
            cells.append(_xlsx_value_cell(f"{_xlsx_column_name(column_index)}{row_index}", value))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in widths.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{columns}</cols><sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def _xlsx_styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="0"/>'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def write_behavior_workbook(
    path: Path,
    *,
    raw_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    model_paths: Sequence[Path],
    boundary_safe_threshold: float,
    vertical_safe_threshold: float,
) -> None:
    """Write the default single-workbook report using only Python's stdlib."""
    manifest_rows: list[list[Any]] = [
        ["item", "value", "filename"],
        ["report", "Paired raw low-level behavior evaluation diagnostics", ""],
        ["core_metrics", "Seven existing metrics retained unchanged", ""],
        ["composite_score", "Not computed", ""],
        ["paired_initial_conditions", "One deep-copied initial Env3DState per scenario/trial", ""],
        ["task_valid", "1 only for outcome=escaped; failed rows remain in raw data", ""],
        ["boundary_safe_threshold", boundary_safe_threshold, "normalized min_boundary_margin"],
        ["vertical_safe_threshold", vertical_safe_threshold, "normalized boundary_margin_z"],
        [],
        ["policy", "checkpoint_full_path", "checkpoint_filename"],
    ]
    for policy, model_path in zip(POLICIES, model_paths):
        resolved = model_path.resolve()
        manifest_rows.append([policy, str(resolved), resolved.name])

    summary_sheet: list[list[Any]] = [
        ["Behavior evaluation summary"],
        ["Raw episode data: behavior_metrics_raw.csv"],
        ["No composite score, weighting, or diagonal pass/fail decision is computed."],
        [],
        *_summary_table_rows(summary_rows),
        [],
        ["Out-of-bounds axis counts (terminal evader position)"],
        *_outcome_axis_rows(raw_rows),
    ]
    sheets: list[tuple[str, list[list[Any]]]] = [
        ("Summary", summary_sheet),
        ("Manifest", manifest_rows),
    ]
    for sheet_name, metrics in SHEET_MATRIX_METRICS.items():
        sheet_rows: list[list[Any]] = [
            [f"{sheet_name} scenario diagnostics"],
            ["Each matrix cell is the episode mean for one scenario × policy pair."],
            [],
        ]
        for metric in metrics:
            sheet_rows.extend([[metric], *_matrix_rows(summary_rows, metric), []])
        if sheet_name == "Boundary":
            sheet_rows.extend([["out_of_bounds_axis_counts"], *_outcome_axis_rows(raw_rows)])
        sheets.append((sheet_name, sheet_rows))

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        + "".join(
            f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            for index, (name, _) in enumerate(sheets, start=1)
        )
        + "</sheets></workbook>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        + "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + "</Types>"
    )
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/styles.xml", _xlsx_styles_xml())
        for index, (_, sheet_rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _xlsx_sheet_xml(sheet_rows))


def write_mean_matrices(out_dir: Path, summary_rows: Sequence[dict[str, Any]]) -> None:
    for metric, filename in MATRIX_FILENAMES.items():
        path = out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", *POLICIES])
            writer.writeheader()
            for scenario in SCENARIOS:
                row: dict[str, Any] = {"scenario": scenario}
                for policy in POLICIES:
                    matches = [
                        item for item in summary_rows
                        if item["scenario"] == scenario and item["policy"] == policy and item["metric"] == metric
                    ]
                    row[policy] = matches[0]["mean"] if matches else ""
                writer.writerow(row)


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired raw low-level behavior metrics (no composite score)")
    parser.add_argument("--episodes", type=int, default=30, help="paired trials per scenario and policy")
    parser.add_argument("--seed", type=int, default=0, help="base seed; shared by all policies within each trial")
    parser.add_argument("--output-dir", default="outputs/paper_eval_core/raw_behavior_metrics")
    parser.add_argument("--boundary-safe-threshold", type=float, default=DEFAULT_BOUNDARY_SAFE_THRESHOLD)
    parser.add_argument("--vertical-safe-threshold", type=float, default=DEFAULT_VERTICAL_SAFE_THRESHOLD)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--policy-checkpoints", nargs=4, default=None, metavar=("PI1", "PI2", "PI3", "PI4"))
    parser.add_argument("--policy1-checkpoint", default="")
    parser.add_argument("--policy2-checkpoint", default="")
    parser.add_argument("--policy3-checkpoint", default="")
    parser.add_argument("--policy4-checkpoint", default="")
    parser.add_argument(
        "--export-matrix-csv",
        action="store_true",
        help="also export the legacy seven matrix CSV files (off by default)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    try:
        _validate_threshold("boundary-safe-threshold", args.boundary_safe_threshold)
        _validate_threshold("vertical-safe-threshold", args.vertical_safe_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    model_paths = resolve_model_paths(args)
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing checkpoint: {path}")
        raise SystemExit("Provide four low-level SAC checkpoints with --policy-checkpoints or --checkpoint-dir.")

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required to load SAC checkpoints.") from exc

    models = [SAC.load(str(path), device=args.device) for path in model_paths]
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial_id in range(args.episodes):
            trial_seed = args.seed + scenario_index * 100_000 + trial_id
            initial_state = sample_paired_initial_state(scenario, trial_seed)
            for policy, model in zip(POLICIES, models):
                rows.append(
                    run_paired_episode(
                        model,
                        scenario=scenario,
                        policy=policy,
                        trial_id=trial_id,
                        seed=trial_seed,
                        initial_state=initial_state,
                        boundary_safe_threshold=args.boundary_safe_threshold,
                        vertical_safe_threshold=args.vertical_safe_threshold,
                    )
                )

    out_dir = Path(args.output_dir)
    write_raw_csv(out_dir / "behavior_metrics_raw.csv", rows)
    summary_rows = summarize_rows(rows, metrics=REPORT_METRICS)
    write_summary_csv(out_dir / "behavior_metrics_summary.csv", summary_rows)
    write_behavior_workbook(
        out_dir / "behavior_evaluation.xlsx",
        raw_rows=rows,
        summary_rows=summary_rows,
        model_paths=model_paths,
        boundary_safe_threshold=args.boundary_safe_threshold,
        vertical_safe_threshold=args.vertical_safe_threshold,
    )
    print(f"[csv] raw: {out_dir / 'behavior_metrics_raw.csv'}")
    print(f"[csv] summary: {out_dir / 'behavior_metrics_summary.csv'}")
    print(f"[xlsx] report: {out_dir / 'behavior_evaluation.xlsx'}")
    if args.export_matrix_csv:
        write_mean_matrices(out_dir, summary_rows)
        print(f"[csv] legacy matrices: {len(MATRIX_FILENAMES)}")
    print("No composite behavior score or diagonal pass/fail decision was computed.")


if __name__ == "__main__":
    main()
