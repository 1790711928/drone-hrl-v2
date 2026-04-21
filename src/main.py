from __future__ import annotations

import argparse

from src.env.pursuit_escape_env import PursuitEscapeEnv


def run_demo(steps: int, scenario: str) -> None:
    env = PursuitEscapeEnv()
    obs = env.reset(scenario=scenario)
    print(f"[reset] scenario={scenario} distance={obs['distance']:.2f}")

    for i in range(steps):
        # beginner-friendly fixed action demo: slightly accelerate + gentle turn
        obs, reward, done, info = env.step((0.2, 0.05, 0.01))
        print(
            f"[step {i+1:03d}] distance={info['distance']:.2f} closing={info['closing_speed']:.2f} "
            f"reward={reward:.3f} outcome={info['outcome']} escape_streak={info['escape_streak']}"
        )
        if done:
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D drone pursuit-escape MVP demo")
    parser.add_argument("--scenario", default="s1_close_threat", help="scenario name")
    parser.add_argument("--steps", type=int, default=30, help="max demo steps")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_demo(steps=args.steps, scenario=args.scenario)
