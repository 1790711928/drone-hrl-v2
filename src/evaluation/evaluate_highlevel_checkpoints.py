"""Evaluate high-level PPO checkpoints on paired deterministic episodes.

This is post-training evaluation only.  It does not modify training, rewards,
low-level models, or checkpoints.  Every checkpoint is evaluated on the same
episode seeds so the resulting success-rate curve is paired across training
progress.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
from pathlib import Path


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)_steps\.zip$", path.name)
    if match is None:
        match = re.search(r"run(\d+).*\.zip$", path.name)
    if match is None:
        raise ValueError(f"checkpoint filename does not contain a step marker: {path.name}")
    return int(match.group(1))


def _collect_checkpoints(checkpoint_dir: Path, pattern: str, max_step: int | None) -> list[Path]:
    paths = sorted(checkpoint_dir.glob(pattern), key=_checkpoint_step)
    if max_step is not None:
        paths = [path for path in paths if _checkpoint_step(path) <= max_step]
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {pattern!r} below {checkpoint_dir}")
    return paths


def _episode_disabled_policy(mode: str, probability: float, seed: int, trial: int) -> int | None:
    if mode == "none":
        return None
    if probability < 1.0 and random.Random(seed * 1_000_003 + trial).random() >= probability:
        return None
    block = trial // 4
    offset = trial % 4
    order = list(range(4))
    random.Random(seed * 10_007 + block).shuffle(order)
    return order[offset]


def _action_probabilities(model, obs) -> list[float]:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor)
    return distribution.distribution.probs.detach().cpu().numpy().reshape(-1).tolist()


def _select_action(
    model,
    obs,
    disabled_policy: int | None,
    fallback_mode: str,
    previous_action: int | None,
    proposed_action=None,
) -> tuple[int, int]:
    if proposed_action is None:
        proposed_action, _ = model.predict(obs, deterministic=True)
    proposed = int(proposed_action)
    if disabled_policy is None or proposed != disabled_policy:
        return proposed, 0
    available = [index for index in range(4) if index != disabled_policy]
    if fallback_mode == "hold_previous" and previous_action in available:
        return int(previous_action), 1
    probabilities = _action_probabilities(model, obs)
    replacement = max(available, key=lambda index: probabilities[index])
    return replacement, 1


def _evaluate_checkpoint(
    model,
    low_models,
    *,
    episodes: int,
    seed: int,
    env_kwargs: dict,
    disable_policy_mode: str,
    disable_probability: float,
    fallback_mode: str,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    from src.training.highlevel_env import HighLevelOptionEnv

    outcomes: list[str] = []
    rewards: list[float] = []
    lengths: list[int] = []
    safe_horizon_successes: list[bool] = []
    early_safe_escapes: list[bool] = []
    final_distances: list[float] = []
    recent_distances: list[float] = []
    recent_closing_speeds: list[float] = []
    completed_phase_counts: list[float] = []
    total_phase_counts: list[float] = []
    raw_rows: list[dict[str, float | int | str]] = []
    env = HighLevelOptionEnv(low_models=low_models, **env_kwargs)
    try:
        for trial in range(episodes):
            episode_seed = seed + trial
            disabled_policy = _episode_disabled_policy(disable_policy_mode, disable_probability, seed, trial)
            obs, _ = env.reset(seed=episode_seed, options={"scenario_set": env_kwargs["scenario_set"]})
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            info: dict = {"outcome": "timeout"}
            previous_action: int | None = None
            remap_count = 0
            requested_actions: list[int] = []
            executed_actions: list[int] = []
            while not (terminated or truncated):
                proposed_action, _ = model.predict(obs, deterministic=True)
                requested = int(proposed_action)
                action, remapped = _select_action(
                    model,
                    obs,
                    disabled_policy,
                    fallback_mode,
                    previous_action,
                    proposed_action=proposed,
                )
                remap_count += remapped
                previous_action = action
                requested_actions.append(requested)
                executed_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            outcomes.append(str(info.get("outcome", "timeout")))
            rewards.append(total_reward)
            lengths.append(steps)
            continuous_steps = int(info.get("continuous_lowlevel_steps", 0))
            target_steps = int(env_kwargs.get("episode_lowlevel_steps", 0))
            safe_horizon_successes.append(
                str(info.get("outcome", "timeout")) == "escaped"
                and target_steps > 0
                and continuous_steps >= target_steps
            )
            early_safe_escapes.append(bool(info.get("early_safe_escape", False)))
            final_distances.append(float(info.get("final_distance", 0.0)))
            recent_distances.append(float(info.get("recent_distance", 0.0)))
            recent_closing_speeds.append(float(info.get("recent_closing_speed", 0.0)))
            completed_phase_counts.append(float(info.get("structured_completed_phases", 0.0)))
            total_phase_counts.append(float(info.get("structured_total_phases", 0.0)))
            raw_rows.append(
                {
                    "trial": trial,
                    "seed": episode_seed,
                    "outcome": str(info.get("outcome", "timeout")),
                    "escaped": int(str(info.get("outcome", "timeout")) == "escaped"),
                    "early_safe_escape": int(bool(info.get("early_safe_escape", False))),
                    "total_reward": total_reward,
                    "episode_length": steps,
                    "final_distance": float(info.get("final_distance", 0.0)),
                    "recent_distance": float(info.get("recent_distance", 0.0)),
                    "recent_closing_speed": float(info.get("recent_closing_speed", 0.0)),
                    "structured_completed_phases": float(info.get("structured_completed_phases", 0.0)),
                    "structured_total_phases": float(info.get("structured_total_phases", 0.0)),
                    "disabled_policy": "none" if disabled_policy is None else f"pi{disabled_policy + 1}",
                    "disabled_policy_index": -1 if disabled_policy is None else disabled_policy,
                    "action_remap_count": remap_count,
                    "requested_actions": ",".join(str(action) for action in requested_actions),
                    "executed_actions": ",".join(str(action) for action in executed_actions),
                }
            )
    finally:
        env.close()

    denominator = max(len(outcomes), 1)
    return {
        "episodes": len(outcomes),
        "escape_streak_required": int(env_kwargs.get("structured_escape_streak", 0)),
        "win_rate": sum(outcome == "escaped" for outcome in outcomes) / denominator,
        "capture_rate": sum(outcome == "captured" for outcome in outcomes) / denominator,
        "out_of_bounds_rate": sum(outcome == "out_of_bounds" for outcome in outcomes) / denominator,
        "timeout_rate": sum(outcome == "timeout" for outcome in outcomes) / denominator,
        "safe_horizon_success_rate": sum(safe_horizon_successes) / denominator,
        "early_safe_escape_rate": sum(early_safe_escapes) / denominator,
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_episode_length": statistics.mean(lengths) if lengths else 0.0,
        "mean_final_distance": statistics.mean(final_distances) if final_distances else 0.0,
        "mean_recent_distance": statistics.mean(recent_distances) if recent_distances else 0.0,
        "mean_recent_closing_speed": statistics.mean(recent_closing_speeds) if recent_closing_speeds else 0.0,
        "mean_structured_completed_phases": statistics.mean(completed_phase_counts) if completed_phase_counts else 0.0,
        "mean_structured_total_phases": statistics.mean(total_phase_counts) if total_phase_counts else 0.0,
        "disable_policy_mode": disable_policy_mode,
        "disable_probability": disable_probability,
        "fallback_mode": fallback_mode,
    }, raw_rows


def _write_plot(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["checkpoint_step"]) for row in rows]
    rates = [100.0 * float(row["win_rate"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, rates, marker="o", linewidth=1.8, label="high-level PPO win rate")
    axis.set_xlabel("training step")
    axis.set_ylabel("win rate (%)")
    axis.set_title("High-level PPO win rate")
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate high-level PPO checkpoint win rates")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-pattern", default="ppo_highlevel_*_steps.zip")
    parser.add_argument("--max-checkpoint-step", type=int, default=None)
    parser.add_argument("--low-model-1", required=True)
    parser.add_argument("--low-model-2", required=True)
    parser.add_argument("--low-model-3", required=True)
    parser.add_argument("--low-model-4", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario-set", default="composite")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    parser.add_argument("--episode-lowlevel-steps", type=int, default=None, help="structured mode defaults to 240; other modes default to 400")
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.20)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    parser.add_argument("--min-regime-hold-steps", type=int, default=20)
    parser.add_argument("--boundary-priority-enter", type=float, default=0.24)
    parser.add_argument("--boundary-priority-exit", type=float, default=0.32)
    parser.add_argument("--showcase-bound-scale", type=float, default=2.5)
    parser.add_argument("--showcase-z-bound-scale", type=float, default=1.5)
    parser.add_argument("--structured-position-jitter", type=float, default=2.0)
    parser.add_argument("--structured-relative-jitter", type=float, default=1.5)
    parser.add_argument("--structured-yaw-jitter", type=float, default=0.18)
    parser.add_argument("--structured-pitch-jitter", type=float, default=0.04)
    parser.add_argument("--structured-speed-jitter", type=float, default=0.5)
    parser.add_argument("--structured-bound-scale", type=float, default=2.5)
    parser.add_argument("--structured-z-bound-scale", type=float, default=1.5)
    parser.add_argument("--success-window-lowlevel-steps", type=int, default=30)
    parser.add_argument("--structured-escape-streak", type=int, default=8)
    parser.add_argument(
        "--structured-four-phase-only",
        action="store_true",
        help="For structured evaluation, sample only the four-phase schedules.",
    )
    parser.add_argument(
        "--structured-phase-count",
        type=int,
        default=None,
        help="For structured evaluation, generate cyclic schedules with this many phases.",
    )
    parser.add_argument(
        "--disable-policy-mode",
        choices=["none", "balanced_random"],
        default="none",
        help="Evaluation-only low-level skill ablation mode.",
    )
    parser.add_argument("--disable-probability", type=float, default=1.0)
    parser.add_argument(
        "--fallback-mode",
        choices=["argmax_available", "hold_previous"],
        default="argmax_available",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_checkpoint_step is not None and args.max_checkpoint_step <= 0:
        parser.error("--max-checkpoint-step must be positive")
    if args.success_window_lowlevel_steps <= 0:
        parser.error("--success-window-lowlevel-steps must be positive")
    if args.structured_escape_streak <= 0:
        parser.error("--structured-escape-streak must be positive")
    if args.structured_phase_count is not None and args.structured_phase_count <= 0:
        parser.error("--structured-phase-count must be positive")
    if args.structured_four_phase_only and args.structured_phase_count not in (None, 4):
        parser.error("--structured-four-phase-only conflicts with --structured-phase-count")
    if not 0.0 <= args.disable_probability <= 1.0:
        parser.error("--disable-probability must be between 0 and 1")
    from src.training.highlevel_env import STRUCTURED_DEFAULT_EPISODE_LOWLEVEL_STEPS

    episode_lowlevel_steps = args.episode_lowlevel_steps
    if episode_lowlevel_steps is None:
        episode_lowlevel_steps = STRUCTURED_DEFAULT_EPISODE_LOWLEVEL_STEPS if args.scenario_set == "structured_random_composite" else 400

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 and gymnasium are required") from exc

    low_paths = [Path(args.low_model_1), Path(args.low_model_2), Path(args.low_model_3), Path(args.low_model_4)]
    missing = [path for path in low_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing low-level model(s): {missing}")
    checkpoints = _collect_checkpoints(Path(args.checkpoint_dir), args.checkpoint_pattern, args.max_checkpoint_step)
    low_models = [SAC.load(str(path), device=args.device) for path in low_paths]
    env_kwargs = {
        "option_duration": args.option_duration,
        "switch_penalty": args.switch_penalty,
        "max_highlevel_steps": args.max_highlevel_steps,
        "scenario_set": args.scenario_set,
        "episode_lowlevel_steps": episode_lowlevel_steps,
        "regime_duration": args.regime_duration,
        "pursuer_speed_ratio": args.pursuer_speed_ratio,
        "regime_schedule": args.regime_schedule,
        "min_regime_hold_steps": args.min_regime_hold_steps,
        "boundary_priority_enter": args.boundary_priority_enter,
        "boundary_priority_exit": args.boundary_priority_exit,
        "showcase_bound_scale": args.showcase_bound_scale,
        "showcase_z_bound_scale": args.showcase_z_bound_scale,
        "structured_position_jitter": args.structured_position_jitter,
        "structured_relative_jitter": args.structured_relative_jitter,
        "structured_yaw_jitter": args.structured_yaw_jitter,
        "structured_pitch_jitter": args.structured_pitch_jitter,
        "structured_speed_jitter": args.structured_speed_jitter,
        "structured_bound_scale": args.structured_bound_scale,
        "structured_z_bound_scale": args.structured_z_bound_scale,
        "success_window_lowlevel_steps": args.success_window_lowlevel_steps,
        "structured_escape_streak": args.structured_escape_streak,
        "structured_four_phase_only": args.structured_four_phase_only,
        "structured_phase_count": args.structured_phase_count,
    }

    rows: list[dict[str, float | int | str]] = []
    raw_rows: list[dict[str, float | int | str]] = []
    for checkpoint in checkpoints:
        model = PPO.load(str(checkpoint), device=args.device)
        result, checkpoint_raw_rows = _evaluate_checkpoint(
            model,
            low_models,
            episodes=args.episodes,
            seed=args.seed,
            env_kwargs=env_kwargs,
            disable_policy_mode=args.disable_policy_mode,
            disable_probability=args.disable_probability,
            fallback_mode=args.fallback_mode,
        )
        for raw_row in checkpoint_raw_rows:
            raw_rows.append({"checkpoint": checkpoint.name, "checkpoint_step": _checkpoint_step(checkpoint), **raw_row})
        row = {"checkpoint": checkpoint.name, "checkpoint_step": _checkpoint_step(checkpoint), **result}
        rows.append(row)
        print(
            f"{checkpoint.name}: win_rate={float(row['win_rate']):.3f}, "
            f"capture={float(row['capture_rate']):.3f}, "
            f"oob={float(row['out_of_bounds_rate']):.3f}, "
            f"timeout={float(row['timeout_rate']):.3f}, "
            f"safe_horizon={float(row['safe_horizon_success_rate']):.3f}, "
            f"early_safe_escape={float(row['early_safe_escape_rate']):.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "highlevel_win_rate.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    raw_csv_path = output_dir / "highlevel_episode_raw.csv"
    raw_fieldnames = list(raw_rows[0]) if raw_rows else ["checkpoint", "checkpoint_step"]
    with raw_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    plot_path = output_dir / "highlevel_win_rate.png"
    _write_plot(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {raw_csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
