"""Low-level SAC training entrypoint for the evader policy."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-level SAC evader policy")
    parser.add_argument("--scenario", default="rear_close_threat")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", default="outputs/checkpoints/sac_lowlevel.zip")
    parser.add_argument("--log-dir", default="outputs/logs/sac")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    from src.training.sac_env import PursuitEscapeGymEnv

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: PursuitEscapeGymEnv(scenario=args.scenario)])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        learning_starts=1_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_out)
    print(f"[lowlevel] SAC training done. model={args.model_out}")


if __name__ == "__main__":
    main()
