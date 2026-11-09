#!/usr/bin/env python3
"""Select and plot representative successful low-level SAC trajectories.

This tool is intentionally independent of the behavior-score and heatmap
pipelines.  It loads four SAC checkpoints only when ``main`` is executed with
real checkpoint paths, rolls out fixed seeds, rejects unsuccessful or invalid
episodes, ranks each policy with a direction-specific heuristic, and writes
candidate metrics, selected trajectories, a manifest, and optional figures.

Local example (run from the repository root)::

    python tools/select_representative_low_level_trajectories.py \
      --checkpoint-dir outputs/checkpoints \
      --seed-start 0 --seed-end 199 \
      --output-dir outputs/representative_low_level_trajectories \
      --deterministic --plot-3d --plot-xoy --add-aircraft

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
    final_velocity = trajectory.evader_velocity[-1] if trajectory.evader_velocity else (0.0, 0.0, 0.0)
    velocity_norm = _norm3(final_velocity)
    if velocity_norm < 1e-8 and len(trajectory.evader) >= 2:
        final_velocity = _sub(trajectory.evader[-1], trajectory.evader[-2])
        velocity_norm = _norm3(final_velocity)
    final_velocity_direction = _scale(final_velocity, 1.0 / velocity_norm) if velocity_norm > 1e-8 else (0.0, 0.0, 0.0)
    first_obs = trajectory.observations[0]
    last_obs = trajectory.observations[-1]
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
    return 2.0 * distance_change + 2.0 * forward_gap + horizontal - 2.5 * vertical


def rank_flank_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer large lateral excursion/path length and reject vertical travel."""
    lateral = _finite_metric(metrics, "lateral_displacement")
    lateral_path = _finite_metric(metrics, "lateral_path_length")
    vertical = _finite_metric(metrics, "vertical_displacement")
    forward = abs(_finite_metric(metrics, "forward_gap_change"))
    return 2.0 * lateral + 1.5 * lateral_path - 2.0 * vertical - 0.5 * forward


def rank_boundary_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer early boundary approach followed by recovery."""
    initial = _finite_metric(metrics, "boundary_margin_initial")
    early_min = _finite_metric(metrics, "boundary_margin_early_min")
    final = _finite_metric(metrics, "boundary_margin_final")
    recovery = _finite_metric(metrics, "boundary_margin_recovery")
    approach = max(0.0, initial - early_min)
    return 3.0 * approach + 3.0 * recovery + 1.5 * final + _finite_metric(metrics, "distance_change")


def rank_vertical_trajectory(metrics: dict[str, Any]) -> float:
    """Prefer sustained vertical ascent and penalize dominant horizontal motion."""
    vertical = _finite_metric(metrics, "vertical_displacement")
    horizontal = _finite_metric(metrics, "horizontal_displacement")
    up_gap = _finite_metric(metrics, "up_gap_change")
    return 3.0 * vertical + 2.0 * up_gap - 2.5 * horizontal


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
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_trajectory_csv(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "evader_x", "evader_y", "evader_z", "pursuer_x", "pursuer_y", "pursuer_z"]
    fields.extend(f"obs_{key}" for key in trajectory.observations[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (evader, pursuer, observation) in enumerate(zip(trajectory.evader, trajectory.pursuer, trajectory.observations)):
            row: dict[str, Any] = {
                "step": index,
                "evader_x": evader[0], "evader_y": evader[1], "evader_z": evader[2],
                "pursuer_x": pursuer[0], "pursuer_y": pursuer[1], "pursuer_z": pursuer[2],
            }
            row.update({f"obs_{key}": value for key, value in observation.items()})
            writer.writerow(row)


def _draw_aircraft(ax: Any, position: tuple[float, float, float], direction: tuple[float, float, float], color: str, scale: float, label: str | None = None) -> None:
    """Draw a small four-rotor aircraft glyph at a trajectory endpoint."""
    norm = _norm3(direction)
    forward = _scale(direction, 1.0 / norm) if norm > 1e-8 else (1.0, 0.0, 0.0)
    up = (0.0, 0.0, 1.0)
    side = (-forward[1], forward[0], 0.0)
    side_norm = _norm3(side)
    if side_norm < 1e-8:
        side = (0.0, 1.0, 0.0)
    else:
        side = _scale(side, 1.0 / side_norm)
    arm_a = _add(_scale(forward, scale), _scale(side, scale))
    arm_b = _add(_scale(forward, scale), _scale(side, -scale))
    for arm in (arm_a, arm_b):
        left = _add(position, arm)
        right = _add(position, _scale(arm, -1.0))
        ax.plot([left[0], right[0]], [left[1], right[1]], [left[2], right[2]], color=color, linewidth=1.5, alpha=0.9)
        ax.scatter([left[0], right[0]], [left[1], right[1]], [left[2], right[2]], color=color, s=12, label=None)
    nose = _add(position, _scale(forward, scale * 1.35))
    ax.plot([position[0], nose[0]], [position[1], nose[1]], [position[2], nose[2]], color=color, linewidth=2.2, label=label)


def plot_strategy_trajectory(trajectory: Trajectory, output_path: Path, *, plot_3d: bool, plot_xoy: bool, add_aircraft: bool) -> None:
    """Save a unified 3D/XOY two-view figure for one selected trajectory."""
    if not plot_3d and not plot_xoy:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = int(plot_3d) + int(plot_xoy)
    fig = plt.figure(figsize=(7.6 * panels, 6.2))
    axes: list[Any] = []
    evader = trajectory.evader
    pursuer = trajectory.pursuer
    final_direction = trajectory.evader_velocity[-1] if trajectory.evader_velocity else _sub(evader[-1], evader[-2])
    all_points = evader + pursuer
    extent = max(max(point[index] for point in all_points) - min(point[index] for point in all_points) for index in range(3))
    aircraft_scale = max(extent * 0.035, 0.25)
    if plot_3d:
        axes.append(fig.add_subplot(1, panels, len(axes) + 1, projection="3d"))
    if plot_xoy:
        axes.append(fig.add_subplot(1, panels, len(axes) + 1))
    for axis in axes:
        if getattr(axis, "name", "") == "3d":
            axis.plot([p[0] for p in evader], [p[1] for p in evader], [p[2] for p in evader], color="red", linewidth=2.2, label="evader")
            axis.plot([p[0] for p in pursuer], [p[1] for p in pursuer], [p[2] for p in pursuer], color="blue", linestyle="--", linewidth=1.5, label="pursuer")
            if add_aircraft:
                _draw_aircraft(axis, evader[-1], final_direction, "red", aircraft_scale, "evader aircraft")
                _draw_aircraft(axis, pursuer[-1], _sub(pursuer[-1], pursuer[-2]), "blue", aircraft_scale, "pursuer aircraft")
            axis.set_zlabel("Z")
            axis.view_init(elev=24, azim=-58)
        else:
            axis.plot([p[0] for p in evader], [p[1] for p in evader], color="red", linewidth=2.2, label="evader")
            axis.plot([p[0] for p in pursuer], [p[1] for p in pursuer], color="blue", linestyle="--", linewidth=1.5, label="pursuer")
            axis.set_aspect("equal", adjustable="datalim")
        if getattr(axis, "name", "") == "3d":
            axis.scatter([evader[0][0]], [evader[0][1]], [evader[0][2]], color="red", marker="o", s=28, label="evader start")
            axis.scatter([pursuer[0][0]], [pursuer[0][1]], [pursuer[0][2]], color="blue", marker="o", s=28, label="pursuer start")
            axis.scatter([evader[-1][0]], [evader[-1][1]], [evader[-1][2]], color="red", marker="^", s=38, label="evader end")
            axis.scatter([pursuer[-1][0]], [pursuer[-1][1]], [pursuer[-1][2]], color="blue", marker="^", s=38, label="pursuer end")
        else:
            axis.scatter([evader[0][0]], [evader[0][1]], color="red", marker="o", s=28, label="evader start")
            axis.scatter([pursuer[0][0]], [pursuer[0][1]], color="blue", marker="o", s=28, label="pursuer start")
            axis.scatter([evader[-1][0]], [evader[-1][1]], color="red", marker="x", s=38, label="evader end")
            axis.scatter([pursuer[-1][0]], [pursuer[-1][1]], color="blue", marker="x", s=38, label="pursuer end")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(alpha=0.25)
        axis.set_title(f"{trajectory.policy_name} | seed={trajectory.seed}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
    return Trajectory(policy_name, scenario, seed, success, outcome, evader, pursuer, observations, velocities)


def _checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    for policy_name, filename in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--{policy_name.replace('_', '-')}-checkpoint", default="", help=f"Override {policy_name} checkpoint (default: checkpoint-dir/{filename})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select representative successful low-level SAC trajectories")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    _checkpoint_arguments(parser)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=199)
    parser.add_argument("--output-dir", default="outputs/representative_low_level_trajectories")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-xoy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-aircraft", action=argparse.BooleanOptionalAction, default=True)
    return parser


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
    selected: dict[str, tuple[Trajectory, dict[str, Any], float]] = {}
    for policy_name, model in models.items():
        for seed in range(args.seed_start, args.seed_end + 1):
            try:
                trajectory = _rollout(model, policy_name, seed, args.deterministic)
                if not trajectory.success:
                    continue
                metrics = trajectory_metrics(trajectory)
                rank_score = RANKERS[policy_name](metrics)
                all_rows.append(_trajectory_row(metrics, rank_score))
                current = selected.get(policy_name)
                if current is None or rank_score > current[2]:
                    selected[policy_name] = (trajectory, metrics, rank_score)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
    if len(selected) != len(POLICY_SCENARIOS):
        raise RuntimeError(f"No valid successful trajectory was found for: {sorted(set(POLICY_SCENARIOS) - set(selected))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_trajectories_metrics.csv", all_rows)
    selected_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"seed_start": args.seed_start, "seed_end": args.seed_end, "deterministic": args.deterministic, "policies": {}}
    for policy_name, (trajectory, metrics, rank_score) in selected.items():
        output_name = POLICY_OUTPUT_NAMES[policy_name]
        trajectory_csv = output_dir / f"{output_name}_seed{trajectory.seed}.csv"
        _save_trajectory_csv(trajectory_csv, trajectory)
        figure_path = output_dir / f"{output_name}_trajectory.png"
        plot_strategy_trajectory(trajectory, figure_path, plot_3d=args.plot_3d, plot_xoy=args.plot_xoy, add_aircraft=args.add_aircraft)
        selected_rows.append({"policy_name": policy_name, "seed": trajectory.seed, "success": trajectory.success, "rank_score": rank_score, "trajectory_csv": str(trajectory_csv), "plot": str(figure_path), **_trajectory_row(metrics, rank_score)})
        manifest["policies"][policy_name] = {"checkpoint": str(checkpoint_paths[policy_name]), "selected_seed": trajectory.seed, "metrics": _trajectory_row(metrics, rank_score), "trajectory_csv": str(trajectory_csv), "plot": str(figure_path), "ranking_rule": RANKERS[policy_name].__name__}
    _write_csv(output_dir / "selected_seeds.csv", selected_rows)
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
