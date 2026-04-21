"""Low-level training entrypoint (SAC for evader in 3D).

Beginner note:
1) Keep high-level selector frozen.
2) Train one SAC policy per scenario first.
3) Save checkpoints before moving to high-level PPO.
"""

from __future__ import annotations

import argparse

from src.env.pursuit_escape_env import PursuitEscapeEnv


def run_smoke_episode(scenario: str, steps: int) -> None:
    env = PursuitEscapeEnv()
    _ = env.reset(scenario=scenario)
    for _ in range(steps):
        _, _, done, _ = env.step((0.1, 0.0, 0.0))
        if done:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="s1_close_threat")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    run_smoke_episode(args.scenario, args.steps)
    print(
        "[lowlevel] smoke run done. Next: integrate SAC library (e.g. Stable-Baselines3) and start actual training."
    )


if __name__ == "__main__":
    main()
