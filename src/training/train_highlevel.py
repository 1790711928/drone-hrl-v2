"""Train the high-level PPO selector with frozen low-level models and logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.training_logging import (
    EpisodeRewardCallback,
    assert_new_output_dir,
    build_run_manifest,
    checkpoint_callback,
    combine_callbacks,
    configure_logger,
    utc_now,
    write_manifest,
)


def _default_low_model_paths() -> list[str]:
    root = Path("outputs/retraining_20261014/formal_lowlevel_50000_seed0")
    return [
        str(root / "pi1_rear/pi1_rear__formal50000__seed0__final.zip"),
        str(root / "pi2_flank/pi2_flank__formal50000__seed0__final.zip"),
        str(root / "pi3_boundary/pi3_boundary__formal50000__seed0__final.zip"),
        str(root / "pi4_vertical/pi4_vertical__formal50000__seed0__final.zip"),
    ]


def _run_label(timesteps: int) -> str:
    if timesteps == 80_000:
        return "formal80000"
    if timesteps == 500:
        return "smoke500"
    return f"run{timesteps}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-level PPO option selector with full run logging")
    default_low = _default_low_model_paths()
    parser.add_argument("--low-model-1", default=default_low[0])
    parser.add_argument("--low-model-2", default=default_low[1])
    parser.add_argument("--low-model-3", default=default_low[2])
    parser.add_argument("--low-model-4", default=default_low[3])
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="outputs/retraining_20261014/formal_highlevel_seed0")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    parser.add_argument(
        "--scenario-set",
        choices=["basic", "mixed", "composite", "sequential", "continuous_pursuit", "continuous_showcase", "scripted_showcase", "structured_random_composite"],
        default="composite",
    )
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
        help="For structured evaluation/training, sample only the four-phase schedules.",
    )
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.checkpoint_freq < 0:
        parser.error("--checkpoint-freq must be non-negative")
    if args.success_window_lowlevel_steps <= 0:
        parser.error("--success-window-lowlevel-steps must be positive")
    if args.structured_escape_streak <= 0:
        parser.error("--structured-escape-streak must be positive")
    from src.training.highlevel_env import HighLevelOptionEnv, STRUCTURED_DEFAULT_EPISODE_LOWLEVEL_STEPS

    episode_lowlevel_steps = args.episode_lowlevel_steps
    if episode_lowlevel_steps is None:
        episode_lowlevel_steps = STRUCTURED_DEFAULT_EPISODE_LOWLEVEL_STEPS if args.scenario_set == "structured_random_composite" else 400

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError("stable-baselines3, gymnasium, and tensorboard are required.") from exc

    low_paths = [Path(args.low_model_1), Path(args.low_model_2), Path(args.low_model_3), Path(args.low_model_4)]
    missing = [path for path in low_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing low-level model(s): {missing}")

    run_dir = Path(args.run_dir)
    assert_new_output_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    final_model = (
        Path(args.model_out)
        if args.model_out
        else run_dir / f"checkpoints/ppo_highlevel__{_run_label(args.timesteps)}__seed{args.seed}__final.zip"
    )
    assert_new_output_dir(final_model)
    monitor_path = run_dir / "monitor.csv"
    reward_csv = run_dir / "episode_rewards.csv"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoints"

    low_models = [SAC.load(str(path), device=args.device) for path in low_paths]

    def make_env():
        base_env = HighLevelOptionEnv(
            low_models=low_models,
            option_duration=args.option_duration,
            switch_penalty=args.switch_penalty,
            max_highlevel_steps=args.max_highlevel_steps,
            scenario_set=args.scenario_set,
            episode_lowlevel_steps=episode_lowlevel_steps,
            regime_duration=args.regime_duration,
            pursuer_speed_ratio=args.pursuer_speed_ratio,
            regime_schedule=args.regime_schedule,
            min_regime_hold_steps=args.min_regime_hold_steps,
            boundary_priority_enter=args.boundary_priority_enter,
            boundary_priority_exit=args.boundary_priority_exit,
            showcase_bound_scale=args.showcase_bound_scale,
            showcase_z_bound_scale=args.showcase_z_bound_scale,
            structured_position_jitter=args.structured_position_jitter,
            structured_relative_jitter=args.structured_relative_jitter,
            structured_yaw_jitter=args.structured_yaw_jitter,
            structured_pitch_jitter=args.structured_pitch_jitter,
            structured_speed_jitter=args.structured_speed_jitter,
            structured_bound_scale=args.structured_bound_scale,
            structured_z_bound_scale=args.structured_z_bound_scale,
            success_window_lowlevel_steps=args.success_window_lowlevel_steps,
            structured_escape_streak=args.structured_escape_streak,
            structured_four_phase_only=args.structured_four_phase_only,
        )
        return Monitor(base_env, filename=str(monitor_path))

    env = DummyVecEnv([make_env])
    env.seed(args.seed)
    logger = configure_logger(run_dir, tensorboard_dir)
    episode_callback = EpisodeRewardCallback(reward_csv, policy_name="highlevel_ppo", seed=args.seed)
    callbacks = combine_callbacks(
        [episode_callback, checkpoint_callback(checkpoint_dir, args.checkpoint_freq, "ppo_highlevel")]
    )
    hyperparameters = {
        "n_steps": 1024,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "option_duration": args.option_duration,
        "switch_penalty": args.switch_penalty,
        "scenario_set": args.scenario_set,
        "episode_lowlevel_steps": episode_lowlevel_steps,
        "regime_duration": args.regime_duration,
        "pursuer_speed_ratio": args.pursuer_speed_ratio,
        "regime_schedule": args.regime_schedule,
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
        "low_model_paths": [str(path.resolve()) for path in low_paths],
    }
    started_at = utc_now()
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        build_run_manifest(
            experiment_name=run_dir.name,
            policy_name="highlevel_ppo",
            status="running",
            seed=args.seed,
            total_timesteps=args.timesteps,
            env=env,
            algorithm="PPO",
            hyperparameters=hyperparameters,
            run_dir=run_dir,
            final_model=final_model,
            reward_csv=reward_csv,
            monitor_csv=monitor_path,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            started_at=started_at,
            finished_at=None,
            callback=episode_callback,
            actual_num_timesteps=0,
        ),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=None,
        device=args.device,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.set_logger(logger)
    status = "failed"
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        final_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(final_model))
        status = "completed"
    finally:
        write_manifest(
            manifest_path,
            build_run_manifest(
                experiment_name=run_dir.name,
                policy_name="highlevel_ppo",
                status=status,
                seed=args.seed,
                total_timesteps=args.timesteps,
                env=env,
                algorithm="PPO",
                hyperparameters=hyperparameters,
                run_dir=run_dir,
                final_model=final_model,
                reward_csv=reward_csv,
                monitor_csv=monitor_path,
                tensorboard_dir=tensorboard_dir,
                checkpoint_dir=checkpoint_dir,
                started_at=started_at,
                finished_at=utc_now(),
                callback=episode_callback,
                actual_num_timesteps=int(model.num_timesteps),
            ),
        )
        env.close()
    print(f"[highlevel] PPO option selector done. model={final_model}")


if __name__ == "__main__":
    main()
