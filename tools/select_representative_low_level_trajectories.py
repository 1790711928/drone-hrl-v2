#!/usr/bin/env python3
"""Scan low-level SAC trajectories and prepare candidates for manual review.

This tool is intentionally independent of the behavior-score and heatmap
pipelines. It loads four SAC checkpoints only when ``main`` is executed with
real checkpoint paths, rolls out fixed seeds, records all candidates, applies
transparent pursuit-quality rejection reasons, and leaves final trajectory
selection to manual review.

Local example (run from the repository root)::

    python tools/select_representative_low_level_trajectories.py \
      --checkpoint-dir outputs/checkpoints \
      --seed-start 0 --seed-end 199 \
      --output-dir outputs/representative_low_level_trajectories \
      --deterministic

The online repository does not contain the user's private checkpoints.  This
module therefore keeps all model imports and rollout work behind ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


# Allow direct execution as ``python tools/<script>.py`` from the repository
# root on Windows and POSIX shells alike.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CHECKPOINTS = {
    "pi1_rear": "sac_low_1_rear_close_threat.zip",
    "pi2_flank": "sac_low_2_flank_threat.zip",
    "pi3_boundary": "sac_low_3_boundary_constrained.zip",
    "pi4_vertical": "sac_low_4_vertical_z_threat.zip",
}
POLICY_SCENARIOS = {
    "pi1_rear": "rear_close_threat",
    "pi2_flank": "flank_threat",
    "pi3_boundary": "boundary_constrained",
    "pi4_vertical": "vertical_z_threat",
}
POLICY_OUTPUT_NAMES = {
    "pi1_rear": "rear_strategy",
    "pi2_flank": "flank_strategy",
    "pi3_boundary": "boundary_strategy",
    "pi4_vertical": "vertical_strategy",
}


@dataclass
class Trajectory:
    policy_name: str
    scenario: str
    seed: int
    success: bool
    outcome: str
    evader: list[tuple[float, float, float]] = field(default_factory=list)
    pursuer: list[tuple[float, float, float]] = field(default_factory=list)
    observations: list[dict[str, float]] = field(default_factory=list)
    evader_velocity: list[tuple[float, float, float]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)


def _norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _scale(vector: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return tuple(component * value for component in vector)  # type: ignore[return-value]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _finite_point(point: tuple[float, float, float]) -> bool:
    return all(math.isfinite(value) for value in point)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _initial_body_axes(trajectory: Trajectory) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return forward/right/up axes from the first observation's attitude."""
    # The evaluator stores yaw/pitch in the observation only as sin/cos and a
    # normalized pitch.  Reconstructing them uses the same definitions as the
    # environment, with pitch_limit=pi/2.
    first = trajectory.observations[0]
    yaw = math.atan2(first["evader_yaw_sin"], first["evader_yaw_cos"])
    pitch = first["evader_pitch"] * (math.pi / 2.0)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    right = (-math.sin(yaw), math.cos(yaw), 0.0)
    up = (-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch))
    return forward, right, up


def trajectory_metrics(trajectory: Trajectory) -> dict[str, Any]:
    """Compute direction-neutral raw metrics for one complete trajectory."""
    if not trajectory.evader or not trajectory.pursuer or not trajectory.observations:
        raise ValueError("trajectory must contain positions and observations")
    if len(trajectory.evader) != len(trajectory.pursuer) or len(trajectory.evader) != len(trajectory.observations):
        raise ValueError("trajectory arrays must have equal lengths")
    if any(not _finite_point(point) for point in trajectory.evader + trajectory.pursuer):
        raise ValueError("trajectory contains a non-finite position")

    initial_evader = trajectory.evader[0]
    final_evader = trajectory.evader[-1]
    initial_relative = _sub(initial_evader, trajectory.pursuer[0])
    final_relative = _sub(final_evader, trajectory.pursuer[-1])
    initial_distance = _norm3(initial_relative)
    final_distance = _norm3(final_relative)
    distance_series = [_norm3(_sub(evader, pursuer)) for evader, pursuer in zip(trajectory.evader, trajectory.pursuer)]
    minimum_distance = min(distance_series)
    time_to_minimum_distance = int(min(range(len(distance_series)), key=distance_series.__getitem__))
    distance_before_escape = distance_series[-2] if trajectory.success and len(distance_series) >= 2 else final_distance
    evader_delta = _sub(final_evader, initial_evader)
    horizontal_displacement = math.hypot(evader_delta[0], evader_delta[1])
    vertical_displacement = abs(evader_delta[2])
    forward_axis, right_axis, up_axis = _initial_body_axes(trajectory)
    forward_positions = [_dot(_sub(point, initial_evader), forward_axis) for point in trajectory.evader]
    right_positions = [_dot(_sub(point, initial_evader), right_axis) for point in trajectory.evader]
    up_positions = [_dot(_sub(point, initial_evader), up_axis) for point in trajectory.evader]
    lateral_displacement = max(right_positions) - min(right_positions)
    lateral_path_length = sum(
        abs(right_positions[index] - right_positions[index - 1]) for index in range(1, len(right_positions))
    )
    trajectory_length = sum(
        _norm3(_sub(trajectory.evader[index], trajectory.evader[index - 1]))
        for index in range(1, len(trajectory.evader))
    )
    boundary_margins = [float(item.get("min_boundary_margin", float("nan"))) for item in trajectory.observations]
    finite_margins = [value for value in boundary_margins if math.isfinite(value)]
    initial_margin = finite_margins[0] if finite_margins else float("nan")
    minimum_margin = min(finite_margins) if finite_margins else float("nan")
    final_margin = finite_margins[-1] if finite_margins else float("nan")
    early_count = max(1, len(finite_margins) // 5)
    early_minimum = min(finite_margins[:early_count]) if finite_margins else float("nan")
    boundary_recovery = final_margin - minimum_margin if finite_margins else float("nan")
    closing_speeds = [float(item.get("closing_speed", float("nan"))) for item in trajectory.observations]
    early_closing = [value for value in closing_speeds[: max(1, len(closing_speeds) * 2 // 5)] if math.isfinite(value)]
    early_distance = _norm3(_sub(trajectory.evader[max(0, len(trajectory.evader) * 2 // 5 - 1)], trajectory.pursuer[max(0, len(trajectory.pursuer) * 2 // 5 - 1)]))
    pursuit_closing_ratio = _safe_mean(1.0 if value >= 0.0 else 0.0 for value in early_closing)
    initial_closing_trend = bool(len(early_closing) >= 2 and pursuit_closing_ratio >= 0.5)
    early_separation_ratio = max(0.0, (early_distance - initial_distance) / max(initial_distance, 1e-6))
    final_velocity = trajectory.evader_velocity[-1] if trajectory.evader_velocity else (0.0, 0.0, 0.0)
    velocity_norm = _norm3(final_velocity)
    if velocity_norm < 1e-8 and len(trajectory.evader) >= 2:
        final_velocity = _sub(trajectory.evader[-1], trajectory.evader[-2])
        velocity_norm = _norm3(final_velocity)
    final_velocity_direction = _scale(final_velocity, 1.0 / velocity_norm) if velocity_norm > 1e-8 else (0.0, 0.0, 0.0)
    first_obs = trajectory.observations[0]
    last_obs = trajectory.observations[-1]
    pursuer_path_length = sum(
        _norm3(_sub(trajectory.pursuer[index], trajectory.pursuer[index - 1]))
        for index in range(1, len(trajectory.pursuer))
    )
    pursuer_valid_steps = sum(_finite_point(point) for point in trajectory.pursuer)
    pursuer_direction = (0.0, 0.0, 0.0)
    if len(trajectory.pursuer) >= 2:
        pursuer_delta = _sub(trajectory.pursuer[-1], trajectory.pursuer[-2])
        pursuer_norm = _norm3(pursuer_delta)
        if pursuer_norm > 1e-8:
            pursuer_direction = _scale(pursuer_delta, 1.0 / pursuer_norm)
    return {
        "policy_name": trajectory.policy_name,
        "scenario": trajectory.scenario,
        "seed": trajectory.seed,
        "success": int(trajectory.success),
        "outcome": trajectory.outcome,
        "capture": int(trajectory.outcome == "captured"),
        "collision": int(trajectory.outcome in {"captured", "out_of_bounds"}),
        "episode_steps": len(trajectory.evader) - 1,
        "initial_position": initial_evader,
        "final_position": final_evader,
        "initial_distance": initial_distance,
        "minimum_distance": minimum_distance,
        "time_to_minimum_distance": time_to_minimum_distance,
        "distance_before_escape": distance_before_escape,
        "final_distance": final_distance,
        "distance_change": final_distance - initial_distance,
        "horizontal_displacement": horizontal_displacement,
        "vertical_displacement": vertical_displacement,
        "lateral_displacement": lateral_displacement,
        "lateral_path_length": lateral_path_length,
        "threat_forward_change": float(last_obs["threat_forward"] - first_obs["threat_forward"]),
        "threat_right_change": float(last_obs["threat_right"] - first_obs["threat_right"]),
        "threat_up_change": float(last_obs["threat_up"] - first_obs["threat_up"]),
        "forward_gap_change": forward_positions[-1] - forward_positions[0],
        "right_gap_change": right_positions[-1] - right_positions[0],
        "up_gap_change": up_positions[-1] - up_positions[0],
        "trajectory_length": trajectory_length,
        "boundary_margin_initial": initial_margin,
        "boundary_margin_min": minimum_margin,
        "boundary_margin_final": final_margin,
        "boundary_margin_recovery": boundary_recovery,
        "boundary_margin_early_min": early_minimum,
        "pursuit_closing_ratio_early": pursuit_closing_ratio,
        "initial_closing_trend": int(initial_closing_trend),
        "pursuit_duration": max(0, len(trajectory.pursuer) - 1),
        "pursuer_path_length": pursuer_path_length,
        "pursuer_valid_steps": pursuer_valid_steps,
        "pursuer_heading_or_velocity_direction": pursuer_direction,
        "pursuer_evader_distance_by_step": ";".join(f"{value:.8g}" for value in distance_series),
        "final_relative_speed": float(last_obs.get("closing_speed", float("nan"))),
        "early_separation_ratio": early_separation_ratio,
        "final_velocity_direction": final_velocity_direction,
    }


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def rank_rear_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer forward-axis/distance expansion and penalize vertical motion."""
    horizontal = _finite_metric(metrics, "horizontal_displacement")
    vertical = _finite_metric(metrics, "vertical_displacement")
    forward_gap = _finite_metric(metrics, "forward_gap_change")
    distance_change = _finite_metric(metrics, "distance_change")
    pursuit_penalty = max(0.0, 0.55 - _finite_metric(metrics, "pursuit_closing_ratio_early")) * 2.0
    early_escape_penalty = max(0.0, _finite_metric(metrics, "early_separation_ratio") - 0.35) * 3.0
    return 2.0 * distance_change + 2.0 * forward_gap + horizontal - 2.5 * vertical - pursuit_penalty - early_escape_penalty


def rank_flank_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer large lateral excursion/path length and reject vertical travel."""
    lateral = _finite_metric(metrics, "lateral_displacement")
    lateral_path = _finite_metric(metrics, "lateral_path_length")
    vertical = _finite_metric(metrics, "vertical_displacement")
    forward = abs(_finite_metric(metrics, "forward_gap_change"))
    pursuit_penalty = max(0.0, 0.55 - _finite_metric(metrics, "pursuit_closing_ratio_early")) * 1.5
    early_escape_penalty = max(0.0, _finite_metric(metrics, "early_separation_ratio") - 0.35) * 2.0
    return 2.0 * lateral + 1.5 * lateral_path - 2.0 * vertical - 0.5 * forward - pursuit_penalty - early_escape_penalty


def rank_boundary_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer early boundary approach followed by recovery."""
    initial = _finite_metric(metrics, "boundary_margin_initial")
    early_min = _finite_metric(metrics, "boundary_margin_early_min")
    final = _finite_metric(metrics, "boundary_margin_final")
    recovery = _finite_metric(metrics, "boundary_margin_recovery")
    approach = max(0.0, initial - early_min)
    pursuit_penalty = max(0.0, 0.55 - _finite_metric(metrics, "pursuit_closing_ratio_early")) * 1.0
    return 3.0 * approach + 3.0 * recovery + 1.5 * final + _finite_metric(metrics, "distance_change") - pursuit_penalty


def rank_vertical_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer sustained vertical ascent and penalize dominant horizontal motion."""
    vertical = _finite_metric(metrics, "vertical_displacement")
    horizontal = _finite_metric(metrics, "horizontal_displacement")
    up_gap = _finite_metric(metrics, "up_gap_change")
    pursuit_penalty = max(0.0, 0.55 - _finite_metric(metrics, "pursuit_closing_ratio_early")) * 1.5
    early_escape_penalty = max(0.0, _finite_metric(metrics, "early_separation_ratio") - 0.35) * 2.0
    return 3.0 * vertical + 2.0 * up_gap - 2.5 * horizontal - pursuit_penalty - early_escape_penalty


RANKERS: dict[str, Callable[[dict[str, Any]], float]] = {
    "pi1_rear": rank_rear_trajectory,
    "pi2_flank": rank_flank_trajectory,
    "pi3_boundary": rank_boundary_trajectory,
    "pi4_vertical": rank_vertical_trajectory,
}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _trajectory_row(metrics: dict[str, Any], rank_score: float) -> dict[str, Any]:
    row = {key: _serialize_value(value) for key, value in metrics.items()}
    row["rank_score"] = rank_score
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_trajectory_csv(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "evader_x", "evader_y", "evader_z", "pursuer_x", "pursuer_y", "pursuer_z",
        "distance", "closing_speed", "threat_forward", "threat_right", "threat_up",
        "boundary_margin", "selected_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (evader, pursuer, observation) in enumerate(zip(trajectory.evader, trajectory.pursuer, trajectory.observations)):
            row: dict[str, Any] = {
                "step": index,
                "evader_x": evader[0], "evader_y": evader[1], "evader_z": evader[2],
                "pursuer_x": pursuer[0], "pursuer_y": pursuer[1], "pursuer_z": pursuer[2],
                "distance": _norm3(_sub(evader, pursuer)),
                "closing_speed": observation.get("closing_speed", ""),
                "threat_forward": observation.get("threat_forward", ""),
                "threat_right": observation.get("threat_right", ""),
                "threat_up": observation.get("threat_up", ""),
                "boundary_margin": observation.get("min_boundary_margin", ""),
                "selected_action": json.dumps(_serialize_value(trajectory.actions[index]), ensure_ascii=False)
                if index < len(trajectory.actions) else "",
            }
            writer.writerow(row)


def plot_strategy_trajectory(trajectory: Trajectory, output_path: Path) -> None:
    """Save one 3D trajectory view; candidate choice remains manual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.2, 6.5))
    axis = fig.add_subplot(111, projection="3d")
    evader = trajectory.evader
    pursuer = trajectory.pursuer
    axis.plot([p[0] for p in evader], [p[1] for p in evader], [p[2] for p in evader], color="red", linewidth=2.2, label="evader")
    axis.plot([p[0] for p in pursuer], [p[1] for p in pursuer], [p[2] for p in pursuer], color="blue", linestyle="--", linewidth=1.5, label="pursuer")
    axis.scatter([evader[0][0]], [evader[0][1]], [evader[0][2]], color="black", marker="x", s=52, linewidths=2.0, label="evader_start")
    axis.scatter([pursuer[0][0]], [pursuer[0][1]], [pursuer[0][2]], color="black", marker="x", s=52, linewidths=2.0, label="pursuer_start")
    axis.scatter([evader[-1][0]], [evader[-1][1]], [evader[-1][2]], color="red", marker="^", s=38, label="evader_end")
    axis.scatter([pursuer[-1][0]], [pursuer[-1][1]], [pursuer[-1][2]], color="blue", marker="^", s=38, label="pursuer_end")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.view_init(elev=24, azim=-58)
    axis.grid(alpha=0.25)
    axis.set_title(f"{trajectory.policy_name} | seed={trajectory.seed}")
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _rollout(model: Any, policy_name: str, seed: int, deterministic: bool) -> Trajectory:
    from src.training.sac_env import PursuitEscapeGymEnv

    scenario = POLICY_SCENARIOS[policy_name]
    env = PursuitEscapeGymEnv(scenario=scenario, randomize_reset=True)
    observation, _ = env.reset(seed=seed)
    evader: list[tuple[float, float, float]] = []
    pursuer: list[tuple[float, float, float]] = []
    observations: list[dict[str, float]] = []
    velocities: list[tuple[float, float, float]] = []
    actions: list[Any] = []
    previous_evader: tuple[float, float, float] | None = None
    outcome = "running"
    try:
        while len(evader) <= env.inner.term_cfg.max_steps:
            state = env.inner.state
            if state is None:
                raise RuntimeError("environment state missing after reset")
            current_evader = (state.evader.x, state.evader.y, state.evader.z)
            current_pursuer = (state.pursuer.x, state.pursuer.y, state.pursuer.z)
            evader.append(current_evader)
            pursuer.append(current_pursuer)
            observations.append({key: float(observation[index]) for index, key in enumerate(PursuitEscapeGymEnv.OBS_KEYS)})
            if previous_evader is None:
                velocities.append((0.0, 0.0, 0.0))
            else:
                velocities.append(_sub(current_evader, previous_evader))
            previous_evader = current_evader
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(_serialize_value(action.tolist() if hasattr(action, "tolist") else action))
            observation, _, terminated, truncated, info = env.step(action)
            outcome = str(info.get("outcome", "running"))
            if terminated or truncated:
                state = env.inner.state
                assert state is not None
                evader.append((state.evader.x, state.evader.y, state.evader.z))
                pursuer.append((state.pursuer.x, state.pursuer.y, state.pursuer.z))
                observations.append({key: float(observation[index]) for index, key in enumerate(PursuitEscapeGymEnv.OBS_KEYS)})
                velocities.append(_sub(evader[-1], evader[-2]))
                break
        success = outcome == "escaped" and len(evader) >= 2
    finally:
        env.close()
    return Trajectory(policy_name, scenario, seed, success, outcome, evader, pursuer, observations, velocities, actions)


def _checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    for policy_name, filename in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--{policy_name.replace('_', '-')}-checkpoint", default="", help=f"Override {policy_name} checkpoint (default: checkpoint-dir/{filename})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan successful low-level SAC trajectory candidates for manual review")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    _checkpoint_arguments(parser)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=199)
    parser.add_argument("--output-dir", default="outputs/representative_low_level_trajectories")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-3d", action=argparse.BooleanOptionalAction, default=False,
                        help="Optionally render every accepted candidate as a 3D review figure; no final seed is selected")
    return parser


def _candidate_rejection_reason(trajectory: Trajectory, metrics: dict[str, Any]) -> str:
    """Return a transparent reason for excluding a rollout from candidate review."""
    if not trajectory.success or trajectory.outcome != "escaped":
        return "outcome_not_escaped"
    if len(trajectory.evader) < 6 or len(trajectory.pursuer) < 6:
        return "pursuer_trajectory_too_short"
    valid_ratio = float(metrics["pursuer_valid_steps"]) / max(1, len(trajectory.pursuer))
    if valid_ratio < 0.95:
        return "pursuer_valid_steps_too_low"
    if not metrics["initial_closing_trend"]:
        return "no_initial_closing_trend"
    if metrics["pursuit_closing_ratio_early"] < 0.35:
        return "pursuit_trend_lost"
    if metrics["early_separation_ratio"] > 0.75:
        return "early_separation_without_pursuit"
    return ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed_end < args.seed_start:
        raise SystemExit("--seed-end must be greater than or equal to --seed-start")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = {
        policy: Path(getattr(args, f"{policy}_checkpoint")) if getattr(args, f"{policy}_checkpoint") else checkpoint_dir / filename
        for policy, filename in DEFAULT_CHECKPOINTS.items()
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for local trajectory selection") from exc

    models = {policy: SAC.load(str(path)) for policy, path in checkpoint_paths.items()}
    all_rows: list[dict[str, Any]] = []
    accepted_count = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    for policy_name, model in models.items():
        for seed in range(args.seed_start, args.seed_end + 1):
            try:
                trajectory = _rollout(model, policy_name, seed, args.deterministic)
                metrics = trajectory_metrics(trajectory)
                rank_score = RANKERS[policy_name](metrics)
                rejection_reason = _candidate_rejection_reason(trajectory, metrics)
                row = _trajectory_row(metrics, rank_score)
                row["candidate_rejection_reason"] = rejection_reason
                row["accepted_candidate"] = int(not rejection_reason)
                all_rows.append(row)
                if trajectory.success:
                    accepted_count += int(not rejection_reason)
                    trajectory_path = trajectory_dir / f"{POLICY_OUTPUT_NAMES[policy_name]}_seed{seed}.csv"
                    _save_trajectory_csv(trajectory_path, trajectory)
                    if args.plot_3d and not rejection_reason:
                        plot_strategy_trajectory(trajectory, output_dir / f"{POLICY_OUTPUT_NAMES[policy_name]}_seed{seed}_3d.png")
            except (RuntimeError, ValueError, FloatingPointError):
                all_rows.append({
                    "policy_name": policy_name, "scenario": POLICY_SCENARIOS[policy_name], "seed": seed,
                    "success": 0, "candidate_rejection_reason": "rollout_or_metric_error", "accepted_candidate": 0,
                })
    _write_csv(output_dir / "candidate_trajectories_metrics.csv", all_rows)
    manifest = {
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "deterministic": args.deterministic,
        "candidate_policy": "all successful rollouts are retained; no automatic final seed selection",
        "accepted_candidate_count": accepted_count,
        "plot_mode": "3D only when --plot-3d is explicitly supplied; aircraft and XOY views are disabled",
        "checkpoints": {policy: str(path) for policy, path in checkpoint_paths.items()},
    }
    (output_dir / "candidate_scan_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
