"""Targeted fine-tuning wrapper for selected low-level specialist policies.

This script intentionally does not alter canonical checkpoints. It loads a single
existing SAC low-level policy, fine-tunes it on one dedicated scenario without
scenario mixing, and writes the result to an explicit output checkpoint path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SUPPORTED_SCENARIOS_BY_POLICY = {
    1: "rear_close_threat",
    4: "vertical_z_threat",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted low-level SAC specialist fine-tune wrapper")
    parser.add_argument("--policy-id", type=int, choices=sorted(SUPPORTED_SCENARIOS_BY_POLICY))
    parser.add_argument("--scenario", choices=sorted(set(SUPPORTED_SCENARIOS_BY_POLICY.values())))
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    expected_scenario = SUPPORTED_SCENARIOS_BY_POLICY[args.policy_id]
    if args.scenario != expected_scenario:
        parser.error(
            f"policy pi{args.policy_id} specialist fine-tune must use scenario {expected_scenario!r}; "
            f"got {args.scenario!r}"
        )

    input_checkpoint = Path(args.input_checkpoint)
    if not input_checkpoint.exists():
        print(f"Missing input checkpoint: {input_checkpoint}")
        print("Please run this command locally with trained low-level SAC checkpoints available.")
        return

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for specialist fine-tuning.") from exc

    from src.training.sac_env import PursuitEscapeGymEnv

    env = DummyVecEnv(
        [
            lambda: PursuitEscapeGymEnv(
                scenario=args.scenario,
                scenario_weights=None,
                randomize_reset=True,
            )
        ]
    )
    env.seed(args.seed)

    print(
        f"[specialist-fix] loading pi{args.policy_id} from {input_checkpoint}; "
        f"scenario={args.scenario}; timesteps={args.timesteps}; seed={args.seed}"
    )
    model = SAC.load(str(input_checkpoint), env=env, device=args.device)
    model.set_random_seed(args.seed)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, progress_bar=True)

    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_checkpoint))
    print(f"[specialist-fix] saved: {output_checkpoint}")


if __name__ == "__main__":
    main()
