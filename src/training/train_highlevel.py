"""Train high-level PPO option selector over four frozen low-level SAC policies."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-level PPO option selector")
    parser.add_argument("--low-model-1", default="outputs/checkpoints/sac_low_1_rear_close_threat.zip")
    parser.add_argument("--low-model-2", default="outputs/checkpoints/sac_low_2_flank_threat.zip")
    parser.add_argument("--low-model-3", default="outputs/checkpoints/sac_low_3_boundary_constrained.zip")
    parser.add_argument("--low-model-4", default="outputs/checkpoints/sac_low_4_vertical_z_threat.zip")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--model-out", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--log-dir", default="", help="optional tensorboard log dir; leave empty to disable")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential", "continuous_pursuit"], default="composite")
    parser.add_argument("--episode-lowlevel-steps", type=int, default=400)
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.20)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    parser.add_argument("--min-regime-hold-steps", type=int, default=20)
    parser.add_argument("--boundary-priority-enter", type=float, default=0.28)
    parser.add_argument("--boundary-priority-exit", type=float, default=0.36)
    args = parser.parse_args()

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.training.highlevel_env import HighLevelOptionEnv

    low_paths = [args.low_model_1, args.low_model_2, args.low_model_3, args.low_model_4]
    for p in low_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing low-level model: {p}")

    low_models = [SAC.load(p, device=args.device) for p in low_paths]

    tensorboard_log = None
    normalized_log_dir = args.log_dir.strip()
    if normalized_log_dir:
        if importlib.util.find_spec("tensorboard") is None:
            print("[highlevel] tensorboard not installed, disabling --log-dir automatically.")
        else:
            Path(normalized_log_dir).mkdir(parents=True, exist_ok=True)
            tensorboard_log = normalized_log_dir

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: HighLevelOptionEnv(
        low_models=low_models,
        option_duration=args.option_duration,
        switch_penalty=args.switch_penalty,
        max_highlevel_steps=args.max_highlevel_steps,
        scenario_set=args.scenario_set,
        episode_lowlevel_steps=args.episode_lowlevel_steps,
        regime_duration=args.regime_duration,
        pursuer_speed_ratio=args.pursuer_speed_ratio,
        regime_schedule=args.regime_schedule,
        min_regime_hold_steps=args.min_regime_hold_steps,
        boundary_priority_enter=args.boundary_priority_enter,
        boundary_priority_exit=args.boundary_priority_exit,
    )])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=args.device,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_out)
    print(f"[highlevel] PPO option selector done. model={args.model_out}")


if __name__ == "__main__":
    main()
