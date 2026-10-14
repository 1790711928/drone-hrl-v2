"""Train one low-level SAC policy with isolated, reproducible run logging."""

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

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]
POLICY_NAMES = {
    "rear_close_threat": "pi1_rear",
    "flank_threat": "pi2_flank",
    "boundary_constrained": "pi3_boundary",
    "vertical_z_threat": "pi4_vertical",
}


def _run_label(timesteps: int) -> str:
    if timesteps == 10_000:
        return "pilot10000"
    if timesteps == 50_000:
        return "formal50000"
    return f"run{timesteps}"


def _default_run_dir(scenario: str, timesteps: int, seed: int) -> Path:
    return Path("outputs/retraining_20261014") / f"manual_lowlevel_{POLICY_NAMES[scenario]}_seed{seed}"


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    policy_name = POLICY_NAMES[args.scenario]
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.model_out:
        run_dir = Path(args.model_out).parent
    else:
        run_dir = _default_run_dir(args.scenario, args.timesteps, args.seed)
    assert_new_output_dir(run_dir)

    if args.model_out:
        final_model = Path(args.model_out)
        assert_new_output_dir(final_model)
    else:
        final_model = run_dir / f"{policy_name}__{_run_label(args.timesteps)}__seed{args.seed}__final.zip"
    return run_dir, final_model, policy_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one low-level SAC policy with full run logging")
    parser.add_argument("--scenario", choices=SCENARIOS, default="rear_close_threat")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="", help="isolated directory under outputs/retraining_20261014")
    parser.add_argument("--model-out", default="", help="optional explicit final model path")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mix-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.checkpoint_freq < 0:
        parser.error("--checkpoint-freq must be non-negative")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError("stable-baselines3, gymnasium, and tensorboard are required.") from exc

    from src.training.sac_env import PursuitEscapeGymEnv

    run_dir, final_model, policy_name = _resolve_paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    monitor_path = run_dir / "monitor.csv"
    reward_csv = run_dir / "episode_rewards.csv"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoints"

    mix_ratio = min(max(args.mix_ratio, 0.0), 0.9)
    scenario_weights = None
    if mix_ratio > 0.0:
        others = [scenario for scenario in SCENARIOS if scenario != args.scenario]
        scenario_weights = {args.scenario: 1.0 - mix_ratio}
        each = mix_ratio / len(others)
        scenario_weights.update({scenario: each for scenario in others})

    def make_env():
        base_env = PursuitEscapeGymEnv(
            scenario=args.scenario,
            scenario_weights=scenario_weights,
            randomize_reset=True,
        )
        return Monitor(base_env, filename=str(monitor_path))

    env = DummyVecEnv([make_env])
    env.seed(args.seed)
    logger = configure_logger(run_dir, tensorboard_dir)
    episode_callback = EpisodeRewardCallback(reward_csv, policy_name=policy_name, seed=args.seed)
    callbacks = combine_callbacks(
        [episode_callback, checkpoint_callback(checkpoint_dir, args.checkpoint_freq, policy_name)]
    )
    hyperparameters = {
        "learning_starts": 1_000,
        "batch_size": 256,
        "train_freq": 1,
        "gradient_steps": 1,
        "mix_ratio": mix_ratio,
        "scenario_weights": scenario_weights,
    }
    started_at = utc_now()
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        build_run_manifest(
            experiment_name=run_dir.name,
            policy_name=policy_name,
            status="running",
            seed=args.seed,
            total_timesteps=args.timesteps,
            env=env,
            algorithm="SAC",
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

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=None,
        device=args.device,
        seed=args.seed,
        learning_starts=1_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
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
                policy_name=policy_name,
                status=status,
                seed=args.seed,
                total_timesteps=args.timesteps,
                env=env,
                algorithm="SAC",
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
    print(f"[lowlevel] SAC training done. model={final_model}")


if __name__ == "__main__":
    main()
