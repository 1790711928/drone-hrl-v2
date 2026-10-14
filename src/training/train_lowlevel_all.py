"""Train all four low-level SAC policies in one isolated experiment directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.training.train_lowlevel import POLICY_NAMES, SCENARIOS, _run_label
from src.training.training_logging import (
    assert_new_output_dir,
    build_experiment_manifest,
    utc_now,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train four low-level SAC policies with isolated logs")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-dir", default="outputs/retraining_20261014/formal_lowlevel_50000_seed0")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--mix-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.checkpoint_freq < 0:
        parser.error("--checkpoint-freq must be non-negative")

    experiment_dir = Path(args.experiment_dir)
    assert_new_output_dir(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name
    planned_runs = [
        {
            "policy_name": POLICY_NAMES[scenario],
            "scenario": scenario,
            "run_dir": str((experiment_dir / POLICY_NAMES[scenario]).resolve()),
            "final_model": str(
                (
                    experiment_dir
                    / POLICY_NAMES[scenario]
                    / f"{POLICY_NAMES[scenario]}__{_run_label(args.timesteps)}__seed{args.seed}__final.zip"
                ).resolve()
            ),
        }
        for scenario in SCENARIOS
    ]
    manifest_path = experiment_dir / "experiment_manifest.json"
    started_at = utc_now()
    write_manifest(
        manifest_path,
        build_experiment_manifest(
            experiment_name=experiment_name,
            status="running",
            seed=args.seed,
            total_timesteps=args.timesteps,
            policy_runs=planned_runs,
            started_at=started_at,
            finished_at=None,
        ),
    )

    status = "failed"
    try:
        for scenario in SCENARIOS:
            policy_name = POLICY_NAMES[scenario]
            run_dir = experiment_dir / policy_name
            command = [
                sys.executable,
                "-m",
                "src.training.train_lowlevel",
                "--scenario",
                scenario,
                "--timesteps",
                str(args.timesteps),
                "--seed",
                str(args.seed),
                "--run-dir",
                str(run_dir),
                "--checkpoint-freq",
                str(args.checkpoint_freq),
                "--mix-ratio",
                str(args.mix_ratio),
                "--device",
                args.device,
            ]
            print(f"[lowlevel-all] training {scenario} -> {run_dir}")
            subprocess.run(command, check=True)
        status = "completed"
    finally:
        write_manifest(
            manifest_path,
            build_experiment_manifest(
                experiment_name=experiment_name,
                status=status,
                seed=args.seed,
                total_timesteps=args.timesteps,
                policy_runs=planned_runs,
                started_at=started_at,
                finished_at=utc_now(),
            ),
        )


if __name__ == "__main__":
    main()
