from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.env.dynamics import Agent3DState
from src.training.highlevel_env import HighLevelOptionEnv, PHASE_SUCCESS_THRESHOLDS


OPTION_NAMES = ("pi1", "pi2", "pi3", "pi4")
PHASE_TYPES = ("rear", "flank", "boundary", "vertical", "rear_vertical")
EXPECTED_OPTIONS: dict[str, tuple[str, ...]] = {
    "rear": ("pi1",),
    "flank": ("pi2",),
    "boundary": ("pi3",),
    "vertical": ("pi4",),
    "rear_vertical": ("pi1", "pi4"),
}
MODEL_FILENAMES = (
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
)
CSV_FIELDS = (
    "phase_type",
    "option",
    "expected_option",
    "episodes",
    "option_duration",
    "phase_success_rate",
    "capture_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "avg_completion_step",
    "avg_reward",
    "avg_distance_gain",
    "avg_boundary_margin_change",
    "avg_threat_right_abs_reduction",
    "avg_vertical_metric",
    "avg_closing_speed_reduction",
)

# Representative evader states only select where a standalone injected phase starts.
# Threat geometry itself remains owned by inject_sequential_phase() in highlevel_env.py.
PHASE_SEED_EVADERS: dict[str, Agent3DState] = {
    "rear": Agent3DState(x=0.0, y=0.0, z=12.0, speed=10.0, yaw=0.0, pitch=0.0),
    "flank": Agent3DState(x=0.0, y=0.0, z=11.0, speed=9.8, yaw=0.0, pitch=0.0),
    "boundary": Agent3DState(x=42.0, y=0.0, z=15.0, speed=9.5, yaw=3.0, pitch=0.0),
    "vertical": Agent3DState(x=0.0, y=0.0, z=16.0, speed=10.0, yaw=0.0, pitch=0.08),
    "rear_vertical": Agent3DState(x=0.0, y=0.0, z=16.0, speed=10.0, yaw=0.0, pitch=0.08),
}


def parse_phase_types(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PHASE_TYPES)
    phases = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(phases) - set(PHASE_TYPES))
    if unknown:
        raise ValueError(f"Unknown phase types: {', '.join(unknown)}")
    if not phases:
        raise ValueError("At least one phase type is required.")
    return phases


def _reset_standalone_phase(env: HighLevelOptionEnv, phase_type: str):
    env.prev_option = None
    env.highlevel_step_count = 0
    env.switch_count = 0
    env.current_scenario_name = f"diagnostic_{phase_type}"
    env.phase_index = 0
    env.completed_phases = 0
    env.phase_names = (phase_type,)
    env.phase_success_by_type = {}
    env.phase_failure_by_type = {}
    return env._inject_current_phase(PHASE_SEED_EVADERS[phase_type])


def _vertical_metric(phase_type: str, obs: dict[str, float]) -> float:
    threshold = PHASE_SUCCESS_THRESHOLDS.get(phase_type, {})
    if "separation_low" not in threshold:
        return 0.0
    separation = abs(obs["dz"])
    is_controlled = (
        threshold["separation_low"] <= separation <= threshold["separation_high"]
        and abs(obs["threat_up"]) <= threshold["threat_up_safe"]
        and obs["boundary_margin_z"] >= threshold["z_margin_safe"]
    )
    return float(is_controlled)


def evaluate_phase_option(
    env: HighLevelOptionEnv,
    phase_type: str,
    option_index: int,
    episodes: int,
) -> dict[str, Any]:
    successes = captures = out_of_bounds = timeouts = 0
    completion_steps: list[int] = []
    rewards: list[float] = []
    distance_gains: list[float] = []
    boundary_margin_changes: list[float] = []
    threat_right_reductions: list[float] = []
    vertical_metrics: list[float] = []
    closing_speed_reductions: list[float] = []

    for _ in range(episodes):
        _reset_standalone_phase(env, phase_type)
        start = env.inner.inner._observation(closing_speed=0.0)
        total_reward = 0.0
        lowlevel_steps = 0
        info: dict[str, Any] = {"outcome": "running"}

        while True:
            _, reward, terminated, truncated, info = env.step(option_index)
            total_reward += float(reward)
            lowlevel_steps += int(info.get("option_duration_used", 0))
            if terminated or truncated:
                break

        final = env.inner.inner._observation(float(info.get("closing_speed", 0.0)))
        outcome = str(info.get("outcome", "timeout"))
        success = bool(info.get("completed_phases", 0) == 1 and outcome == "escaped")
        successes += int(success)
        captures += int(outcome == "captured")
        out_of_bounds += int(outcome == "out_of_bounds")
        timeouts += int(outcome == "timeout")
        if success:
            completion_steps.append(lowlevel_steps)

        rewards.append(total_reward)
        distance_gains.append(final["distance"] - start["distance"])
        boundary_margin_changes.append(final["min_boundary_margin"] - start["min_boundary_margin"])
        threat_right_reductions.append(abs(start["threat_right"]) - abs(final["threat_right"]))
        vertical_metrics.append(_vertical_metric(phase_type, final))
        closing_speed_reductions.append(start["closing_speed"] - final["closing_speed"])

    return {
        "phase_type": phase_type,
        "option": OPTION_NAMES[option_index],
        "expected_option": "/".join(EXPECTED_OPTIONS[phase_type]),
        "episodes": episodes,
        "option_duration": env.option_duration,
        "phase_success_rate": successes / episodes,
        "capture_rate": captures / episodes,
        "out_of_bounds_rate": out_of_bounds / episodes,
        "timeout_rate": timeouts / episodes,
        "avg_completion_step": mean(completion_steps) if completion_steps else 0.0,
        "avg_reward": mean(rewards),
        "avg_distance_gain": mean(distance_gains),
        "avg_boundary_margin_change": mean(boundary_margin_changes),
        "avg_threat_right_abs_reduction": mean(threat_right_reductions),
        "avg_vertical_metric": mean(vertical_metrics),
        "avg_closing_speed_reduction": mean(closing_speed_reductions),
    }


def print_diagnostics(rows: list[dict[str, Any]], phases: list[str]) -> None:
    by_phase = {phase: [row for row in rows if row["phase_type"] == phase] for phase in phases}
    print("\n=== Phase × Option Success Matrix ===")
    for phase in phases:
        scores = {str(row["option"]): float(row["phase_success_rate"]) for row in by_phase[phase]}
        print(f"{phase} -> " + ", ".join(f"{option}:{scores[option]:.3f}" for option in OPTION_NAMES))

    print("\n=== Per-phase Discriminability ===")
    top_options: list[str] = []
    expected_scores: list[float] = []
    wrong_scores: list[float] = []
    ambiguous_phases: list[str] = []
    for phase in phases:
        scores = {str(row["option"]): float(row["phase_success_rate"]) for row in by_phase[phase]}
        best_score = max(scores.values())
        best_options = [option for option in OPTION_NAMES if scores[option] == best_score]
        best_option = "/".join(best_options)
        top_options.extend(best_options)
        expected = EXPECTED_OPTIONS[phase]
        expected_score = max(scores[option] for option in expected)
        best_wrong_score = max(scores[option] for option in OPTION_NAMES if option not in expected)
        expected_rank = 1 + sum(score > expected_score for score in scores.values())
        margin = expected_score - best_wrong_score
        warning = ""
        if expected_score < best_score:
            warning = " WARNING: expected option is not top-1"
        if best_score - min(scores.values()) <= 0.10:
            ambiguous_phases.append(phase)
        expected_scores.append(expected_score)
        wrong_scores.append(best_wrong_score)
        print(
            f"{phase}: best_option={best_option}, expected_option={'/'.join(expected)}, "
            f"expected_option_rank={expected_rank}, expected_minus_second={margin:+.3f}{warning}"
        )

    top_counts = Counter(top_options)
    collapse = [option for option, count in top_counts.items() if count > len(phases) / 2]
    collapse_warning = (
        "none" if not collapse else f"possible generic option collapse: {', '.join(sorted(collapse))} top-1 in most phases"
    )
    ambiguity_warning = "none" if not ambiguous_phases else f"ambiguous phases: {', '.join(ambiguous_phases)}"
    print("\n=== Overall Discriminability ===")
    print(f"mean_expected_option_success={mean(expected_scores):.3f}")
    print(f"mean_best_wrong_option_success={mean(wrong_scores):.3f}")
    print(f"option_collapse_warning={collapse_warning}")
    print(f"phase_ambiguity_warning={ambiguity_warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phase × option discriminability without training PPO")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--phase-types", default="all", help="Comma-separated rear,flank,boundary,vertical,rear_vertical or all")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.option_duration <= 0:
        parser.error("--option-duration must be positive")
    try:
        phases = parse_phase_types(args.phase_types)
    except ValueError as exc:
        parser.error(str(exc))

    checkpoint_dir = Path(args.checkpoint_dir)
    model_paths = [checkpoint_dir / filename for filename in MODEL_FILENAMES]
    for path in model_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    from stable_baselines3 import SAC

    models = [SAC.load(str(path)) for path in model_paths]
    env = HighLevelOptionEnv(low_models=models, option_duration=args.option_duration, scenario_set="sequential")
    rows = [
        evaluate_phase_option(env, phase_type, option_index, args.episodes)
        for phase_type in phases
        for option_index in range(len(OPTION_NAMES))
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "phase_option_discriminability.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_diagnostics(rows, phases)
    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
