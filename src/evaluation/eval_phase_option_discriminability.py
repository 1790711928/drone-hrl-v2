from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.env.dynamics import Agent3DState, Env3DState
from src.env.scenarios import SCENARIOS
from src.training.highlevel_env import HighLevelOptionEnv, PHASE_SUCCESS_THRESHOLDS


OPTION_NAMES = ("pi1", "pi2", "pi3", "pi4")
PHASE_TYPES = ("rear", "flank", "boundary", "vertical", "rear_vertical")
EVAL_MODES = ("one_shot", "sustained")
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
    "eval_mode",
    "phase_type",
    "option",
    "expected_option",
    "episodes",
    "option_duration",
    "phase_success_rate",
    "one_shot_success_rate",
    "sustained_success_rate",
    "improvement_score",
    "distance_gain",
    "closing_speed_reduction",
    "rear_pressure_abs_reduction",
    "rear_pressure_improvement_score",
    "threat_right_abs_reduction",
    "lateral_threat_improvement_score",
    "min_boundary_margin_change",
    "boundary_recovery_improvement_score",
    "vertical_target_band_rate",
    "controlled_z_margin_rate",
    "threat_up_abs_reduction",
    "z_margin_change",
    "vertical_improvement_score",
    "rear_vertical_combined_score",
    "capture_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "avg_reward",
    "avg_completion_step",
)

# These representative evader states only select where a standalone injected
# phase starts. Threat geometry and phase thresholds remain owned by
# highlevel_env.py, so this diagnostic cannot silently tune the benchmark.
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


def expand_eval_modes(raw: str) -> list[str]:
    return list(EVAL_MODES) if raw == "both" else [raw]


def _prepare_standalone_phase(env: HighLevelOptionEnv, phase_type: str, source: str) -> None:
    env.prev_option = None
    env.highlevel_step_count = 0
    env.switch_count = 0
    env.current_scenario_name = f"diagnostic_{source}_{phase_type}"
    env.phase_index = 0
    env.completed_phases = 0
    env.phase_names = (phase_type,)
    env.phase_success_by_type = {}
    env.phase_failure_by_type = {}


def reset_injected_phase(env: HighLevelOptionEnv, phase_type: str) -> None:
    _prepare_standalone_phase(env, phase_type, "injected")
    env._inject_current_phase(PHASE_SEED_EVADERS[phase_type])


def reset_canonical_phase(env: HighLevelOptionEnv, phase_type: str) -> None:
    if phase_type == "rear_vertical":
        reset_injected_phase(env, phase_type)
        return
    scenario_by_phase = {
        "rear": "rear_close_threat",
        "flank": "flank_threat",
        "boundary": "boundary_constrained",
        "vertical": "vertical_z_threat",
    }
    scenario = scenario_by_phase[phase_type]
    spec = SCENARIOS[scenario]
    _prepare_standalone_phase(env, phase_type, "canonical")
    env._set_inner_state(Env3DState(evader=spec.evader, pursuer=spec.pursuer, step_count=0), scenario)
    env._reset_phase_tracking()


def _estimated_closing_speed(env: HighLevelOptionEnv) -> float:
    state = env.inner.inner.state
    assert state is not None
    ev, pu = state.evader, state.pursuer
    rel = (ev.x - pu.x, ev.y - pu.y, ev.z - pu.z)
    rel_norm = max(math.sqrt(sum(value * value for value in rel)), 1e-6)

    def velocity(agent: Agent3DState) -> tuple[float, float, float]:
        return (
            agent.speed * math.cos(agent.pitch) * math.cos(agent.yaw),
            agent.speed * math.cos(agent.pitch) * math.sin(agent.yaw),
            agent.speed * math.sin(agent.pitch),
        )

    ev_velocity, pu_velocity = velocity(ev), velocity(pu)
    distance_rate = sum(rel[index] * (ev_velocity[index] - pu_velocity[index]) for index in range(3)) / rel_norm
    return -distance_rate


def _current_observation(env: HighLevelOptionEnv, info: dict[str, Any] | None = None) -> dict[str, float]:
    raw_closing_speed = float(info["closing_speed"]) if info and "closing_speed" in info else _estimated_closing_speed(env)
    return env.inner.inner._observation(raw_closing_speed)


def _vertical_target_metric(phase_type: str, obs: dict[str, float]) -> float:
    threshold = PHASE_SUCCESS_THRESHOLDS.get(phase_type, {})
    if "separation_low" not in threshold:
        return 0.0
    separation = abs(obs["dz"])
    controlled = (
        threshold["separation_low"] <= separation <= threshold["separation_high"]
        and abs(obs["threat_up"]) <= threshold["threat_up_safe"]
        and obs["boundary_margin_z"] >= threshold["z_margin_safe"]
    )
    return float(controlled)


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _improvement_metrics(phase_type: str, start: dict[str, float], final: dict[str, float]) -> dict[str, float]:
    distance_gain = final["distance"] - start["distance"]
    closing_speed_reduction = start["closing_speed"] - final["closing_speed"]
    rear_pressure_reduction = abs(start["threat_forward"]) - abs(final["threat_forward"])
    threat_right_reduction = abs(start["threat_right"]) - abs(final["threat_right"])
    boundary_margin_change = final["min_boundary_margin"] - start["min_boundary_margin"]
    target_band = _vertical_target_metric(phase_type, final)
    controlled_z_margin = float(final["boundary_margin_z"] >= PHASE_SUCCESS_THRESHOLDS.get(phase_type, {}).get("z_margin_safe", 0.20))
    threat_up_reduction = abs(start["threat_up"]) - abs(final["threat_up"])
    z_margin_change = final["boundary_margin_z"] - start["boundary_margin_z"]

    # Bounded, phase-specific scores prevent unrelated distance or margin changes
    # from dominating the diagnostic. These values rank options only and never
    # feed environment rewards or training logic.
    rear_score = (
        0.40 * _clip_unit(distance_gain / 0.025)
        + 0.35 * _clip_unit(closing_speed_reduction / 0.08)
        + 0.25 * _clip_unit(rear_pressure_reduction / 0.20)
    )
    lateral_score = 0.85 * _clip_unit(threat_right_reduction / 0.20) + 0.15 * _clip_unit(distance_gain / 0.02)
    boundary_score = _clip_unit(boundary_margin_change / 0.07)
    vertical_score = (
        0.35 * target_band
        + 0.30 * controlled_z_margin
        + 0.25 * _clip_unit(threat_up_reduction / 0.20)
        + 0.10 * _clip_unit(z_margin_change / 0.10)
    )
    combined_score = 0.40 * rear_score + 0.40 * vertical_score + 0.20 * min(rear_score, vertical_score)
    score_by_phase = {
        "rear": rear_score,
        "flank": lateral_score,
        "boundary": boundary_score,
        "vertical": vertical_score,
        "rear_vertical": combined_score,
    }
    return {
        "distance_gain": distance_gain,
        "closing_speed_reduction": closing_speed_reduction,
        "rear_pressure_abs_reduction": rear_pressure_reduction,
        "rear_pressure_improvement_score": rear_score,
        "threat_right_abs_reduction": threat_right_reduction,
        "lateral_threat_improvement_score": lateral_score,
        "min_boundary_margin_change": boundary_margin_change,
        "boundary_recovery_improvement_score": boundary_score,
        "vertical_target_band_rate": target_band,
        "controlled_z_margin_rate": controlled_z_margin,
        "threat_up_abs_reduction": threat_up_reduction,
        "z_margin_change": z_margin_change,
        "vertical_improvement_score": vertical_score,
        "rear_vertical_combined_score": combined_score,
        "improvement_score": score_by_phase[phase_type],
    }


def evaluate_phase_option(
    env: HighLevelOptionEnv,
    phase_type: str,
    option_index: int,
    episodes: int,
    eval_mode: str,
    source: str = "injected",
) -> dict[str, Any]:
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")
    successes = captures = out_of_bounds = timeouts = 0
    completion_steps: list[int] = []
    rewards: list[float] = []
    metric_samples: list[dict[str, float]] = []

    for _ in range(episodes):
        if source == "canonical":
            reset_canonical_phase(env, phase_type)
        elif source == "injected":
            reset_injected_phase(env, phase_type)
        else:
            raise ValueError(f"Unsupported phase source: {source}")
        start = _current_observation(env)
        total_reward = 0.0
        lowlevel_steps = 0
        info: dict[str, Any] = {"outcome": "running"}

        while True:
            _, reward, terminated, truncated, info = env.step(option_index)
            total_reward += float(reward)
            lowlevel_steps += int(info.get("option_duration_used", 0))
            if eval_mode == "one_shot" or terminated or truncated:
                break

        final = _current_observation(env, info)
        outcome = str(info.get("outcome", "running"))
        success = bool(info.get("completed_phases", 0) == 1 and outcome == "escaped")
        successes += int(success)
        captures += int(outcome == "captured")
        out_of_bounds += int(outcome == "out_of_bounds")
        timeouts += int(outcome == "timeout")
        if success:
            completion_steps.append(lowlevel_steps)

        metrics = _improvement_metrics(phase_type, start, final)
        if outcome == "out_of_bounds":
            metrics["improvement_score"] -= 2.0
        elif outcome == "captured":
            metrics["improvement_score"] -= 1.0
        metric_samples.append(metrics)
        rewards.append(total_reward)

    averaged_metrics = {key: mean(sample[key] for sample in metric_samples) for key in metric_samples[0]}
    success_rate = successes / episodes
    return {
        "eval_mode": eval_mode,
        "phase_type": phase_type,
        "option": OPTION_NAMES[option_index],
        "expected_option": "/".join(EXPECTED_OPTIONS[phase_type]),
        "episodes": episodes,
        "option_duration": env.option_duration,
        "phase_success_rate": success_rate,
        "one_shot_success_rate": success_rate if eval_mode == "one_shot" else "",
        "sustained_success_rate": success_rate if eval_mode == "sustained" else "",
        **averaged_metrics,
        "capture_rate": captures / episodes,
        "out_of_bounds_rate": out_of_bounds / episodes,
        "timeout_rate": timeouts / episodes,
        "avg_reward": mean(rewards),
        "avg_completion_step": mean(completion_steps) if completion_steps else 0.0,
    }


def print_diagnostics(rows: list[dict[str, Any]], phases: list[str], eval_mode: str) -> None:
    by_phase = {phase: [row for row in rows if row["phase_type"] == phase] for phase in phases}
    print(f"\n=== Phase × Option Improvement Matrix ({eval_mode}) ===")
    for phase in phases:
        scores = {str(row["option"]): float(row["improvement_score"]) for row in by_phase[phase]}
        print(f"{phase} -> " + ", ".join(f"{option}:{scores[option]:+.3f}" for option in OPTION_NAMES))

    print(f"\n=== Phase × Option Success Matrix ({eval_mode}) ===")
    for phase in phases:
        scores = {str(row["option"]): float(row["phase_success_rate"]) for row in by_phase[phase]}
        print(f"{phase} -> " + ", ".join(f"{option}:{scores[option]:.3f}" for option in OPTION_NAMES))

    print(f"\n=== Per-phase Discriminability by Improvement Score ({eval_mode}) ===")
    top_options: list[str] = []
    expected_scores: list[float] = []
    wrong_scores: list[float] = []
    ambiguous_phases: list[str] = []
    for phase in phases:
        scores = {str(row["option"]): float(row["improvement_score"]) for row in by_phase[phase]}
        best_score = max(scores.values())
        best_options = [option for option in OPTION_NAMES if scores[option] == best_score]
        top_options.extend(best_options)
        expected = EXPECTED_OPTIONS[phase]
        expected_score = max(scores[option] for option in expected)
        best_wrong_score = max(scores[option] for option in OPTION_NAMES if option not in expected)
        expected_rank = 1 + sum(score > expected_score for score in scores.values())
        margin = expected_score - best_wrong_score
        warning = " WARNING: expected option is not top-1" if expected_score < best_score else ""
        if best_score - min(scores.values()) <= 0.10:
            ambiguous_phases.append(phase)
        expected_scores.append(expected_score)
        wrong_scores.append(best_wrong_score)
        print(
            f"{phase}: best_option_by_score={'/'.join(best_options)}, expected_option={'/'.join(expected)}, "
            f"expected_option_rank={expected_rank}, expected_minus_second={margin:+.3f}{warning}"
        )

    top_counts = Counter(top_options)
    collapse = [option for option, count in top_counts.items() if count > len(phases) / 2]
    collapse_warning = "none" if not collapse else f"possible generic option collapse: {', '.join(sorted(collapse))} top-1 in most phases"
    ambiguity_warning = "none" if not ambiguous_phases else f"ambiguous phases: {', '.join(ambiguous_phases)}"
    print(f"\n=== Overall Discriminability by Improvement Score ({eval_mode}) ===")
    print(f"mean_expected_option_improvement_score={mean(expected_scores):+.3f}")
    print(f"mean_best_wrong_option_improvement_score={mean(wrong_scores):+.3f}")
    print(f"option_collapse_warning={collapse_warning}")
    print(f"phase_ambiguity_warning={ambiguity_warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-shot and sustained phase × option discriminability without training PPO")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--phase-types", default="all", help="Comma-separated rear,flank,boundary,vertical,rear_vertical or all")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--eval-mode", choices=["one_shot", "sustained", "both"], default="one_shot")
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
    rows: list[dict[str, Any]] = []
    for eval_mode in expand_eval_modes(args.eval_mode):
        mode_rows = [
            evaluate_phase_option(env, phase_type, option_index, args.episodes, eval_mode)
            for phase_type in phases
            for option_index in range(len(OPTION_NAMES))
        ]
        rows.extend(mode_rows)
        print_diagnostics(mode_rows, phases, eval_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "phase_option_discriminability.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
