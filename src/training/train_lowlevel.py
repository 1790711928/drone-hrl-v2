"""Low-level SAC training entrypoint for the evader policy."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-level SAC evader policy")
    parser.add_argument("--scenario", default="rear_close_threat")
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--model-out", default="outputs/checkpoints/sac_lowlevel.zip")
    parser.add_argument("--log-dir", default="", help="optional tensorboard log dir; leave empty to disable")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.2,
        help="portion sampled from non-primary scenarios (0.0 disables mixing)",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    from src.training.sac_env import PursuitEscapeGymEnv

    tensorboard_log = None
    normalized_log_dir = args.log_dir.strip()
    if normalized_log_dir:
        if importlib.util.find_spec("tensorboard") is None:
            print("[lowlevel] tensorboard not installed, disabling --log-dir automatically.")
        else:
            Path(normalized_log_dir).mkdir(parents=True, exist_ok=True)
            tensorboard_log = normalized_log_dir

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    primary = args.scenario
    if primary == "flank_encirclement":
        primary = "flank_threat"

    mix_ratio = min(max(args.mix_ratio, 0.0), 0.9)
    if primary not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {primary}. available={SCENARIOS}")

    scenario_weights: dict[str, float] | None = None
    if mix_ratio > 0.0:
        others = [s for s in SCENARIOS if s != primary]
        scenario_weights = {primary: 1.0 - mix_ratio}
        if others:
            each = mix_ratio / len(others)
            for s in others:
                scenario_weights[s] = each
        print(f"[lowlevel] scenario mix for {primary}: {scenario_weights}")

    env = DummyVecEnv(
        [
            lambda: PursuitEscapeGymEnv(
                scenario=primary,
                scenario_weights=scenario_weights,
                randomize_reset=True,
            )
        ]
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log,
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
