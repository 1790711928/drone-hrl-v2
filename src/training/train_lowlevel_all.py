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
    parser.add_argument("--log-dir", default="", help="optional tensorboard log dir; leave empty to disable")
    args = parser.parse_args()

    print(f"[lowlevel-all] python executable: {sys.executable}")

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
        if args.log_dir:
            cmd += ["--log-dir", f"{args.log_dir}/{scenario}"]

        print(f"[lowlevel-all] training {scenario} -> {model_out}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "lowlevel-all failed. Ensure dependencies are installed: "
                "pip install -r requirements.txt (or at least stable-baselines3 gymnasium). "
                "If you enable --log-dir and hit tensorboard errors, run: pip install tensorboard"
            ) from exc


if __name__ == "__main__":
    main()
