from __future__ import annotations

import argparse
import subprocess
import sys

SCENARIOS = [
    "rear_close_threat",
    "flank_encirclement",
    "boundary_constrained",
    "vertical_z_threat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train four low-level SAC policies (one per scenario)")
    parser.add_argument("--timesteps", type=int, default=120_000)
    args = parser.parse_args()

    for idx, scenario in enumerate(SCENARIOS, start=1):
        model_out = f"outputs/checkpoints/sac_low_{idx}_{scenario}.zip"
        cmd = [
            sys.executable,
            "-m",
            "src.training.train_lowlevel",
            "--scenario",
            scenario,
            "--timesteps",
            str(args.timesteps),
            "--model-out",
            model_out,
        ]
        print(f"[lowlevel-all] training {scenario} -> {model_out}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
