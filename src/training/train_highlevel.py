"""Train high-level PPO switcher over four frozen low-level SAC policies."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-level PPO switch policy")
    parser.add_argument("--low-model-1", default="outputs/checkpoints/sac_low_1_rear_close_threat.zip")
    parser.add_argument("--low-model-2", default="outputs/checkpoints/sac_low_2_flank_encirclement.zip")
    parser.add_argument("--low-model-3", default="outputs/checkpoints/sac_low_3_boundary_constrained.zip")
    parser.add_argument("--low-model-4", default="outputs/checkpoints/sac_low_4_vertical_z_threat.zip")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--model-out", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--log-dir", default="outputs/logs/ppo_high")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    from src.training.highlevel_env import HighLevelSwitchEnv

    low_paths = [args.low_model_1, args.low_model_2, args.low_model_3, args.low_model_4]
    for p in low_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing low-level model: {p}")

    low_models = [SAC.load(p, device=args.device) for p in low_paths]

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: HighLevelSwitchEnv(low_models=low_models)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_out)
    print(f"[highlevel] PPO switcher done. model={args.model_out}")


if __name__ == "__main__":
    main()
