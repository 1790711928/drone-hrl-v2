from __future__ import annotations

import argparse
from dataclasses import dataclass


SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]


@dataclass
class EvalResult:
    success_rate: float
    avg_reward: float


def evaluate_model_on_scenario(model, scenario: str, episodes: int):
    from src.training.sac_env import PursuitEscapeGymEnv

    successes = 0
    rewards = []
    for _ in range(episodes):
        env = PursuitEscapeGymEnv(scenario=scenario)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        if info.get("outcome") == "escaped":
            successes += 1
        rewards.append(ep_reward)

    return EvalResult(success_rate=successes / max(episodes, 1), avg_reward=sum(rewards) / max(len(rewards), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="4x4 evaluation for low-level SAC policies")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--model-1", default="outputs/checkpoints/sac_low_1_rear_close_threat.zip")
    parser.add_argument("--model-2", default="outputs/checkpoints/sac_low_2_flank_threat.zip")
    parser.add_argument("--model-3", default="outputs/checkpoints/sac_low_3_boundary_constrained.zip")
    parser.add_argument("--model-4", default="outputs/checkpoints/sac_low_4_vertical_z_threat.zip")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    models = [
        SAC.load(args.model_1),
        SAC.load(args.model_2),
        SAC.load(args.model_3),
        SAC.load(args.model_4),
    ]

    print("=== Low-level 4x4 Success Matrix ===")
    for i, scenario in enumerate(SCENARIOS, start=1):
        row = []
        for j, model in enumerate(models, start=1):
            r = evaluate_model_on_scenario(model, scenario, args.episodes)
            row.append(f"pi{j}:sr={r.success_rate:.2f},R={r.avg_reward:.1f}")
        print(f"S{i}({scenario}) -> " + " | ".join(row))

    print("\nCriterion: each pi_i should be best (or tied-best) on S_i before high-level PPO training.")


if __name__ == "__main__":
    main()
