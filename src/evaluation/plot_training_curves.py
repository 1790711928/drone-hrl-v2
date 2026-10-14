"""Create paper-ready training-curve figures from locally collected CSV logs.

This utility is intentionally post-processing only.  It never loads a model,
creates an environment, or starts training.  The raw episode data remain in
``episode_rewards.csv``; generated PNG/CSV files are disposable plot views.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "episode_index": float(row["episode_index"]),
                    "global_step": float(row["global_step"]),
                    "episode_return": float(row["episode_return"]),
                    "episode_length": float(row["episode_length"]),
                }
            )
        return rows


def _moving_average(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        output.append(running / min(index + 1, window))
    return output


def _write_plot_data(path: Path, rows: list[dict[str, float]], window: int) -> None:
    returns = [row["episode_return"] for row in rows]
    moving = _moving_average(returns, window)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "global_step",
                "episode_return",
                "moving_average",
                "episode_length",
            ],
        )
        writer.writeheader()
        for row, average in zip(rows, moving):
            writer.writerow({**row, "moving_average": average})


def _plot_policy(policy_dir: Path, output_dir: Path, window: int, plt) -> tuple[str, list[dict[str, float]]]:
    reward_path = policy_dir / "episode_rewards.csv"
    rows = _read_rows(reward_path)
    if not rows:
        raise ValueError(f"episode_rewards.csv has no data rows: {reward_path}")
    policy_name = policy_dir.name
    _write_plot_data(output_dir / f"{policy_name}_plot_data.csv", rows, window)
    steps = [row["global_step"] for row in rows]
    returns = [row["episode_return"] for row in rows]
    moving = _moving_average(returns, window)
    lengths = [row["episode_length"] for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, returns, alpha=0.28, linewidth=0.8, label="episode return")
    axis.plot(steps, moving, linewidth=1.8, label=f"moving average ({window})")
    axis.set_xlabel("environment step")
    axis.set_ylabel("episode reward")
    axis.set_title(f"{policy_name}: training reward")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{policy_name}_reward_curve.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, lengths, linewidth=1.2)
    axis.set_xlabel("environment step")
    axis.set_ylabel("episode length")
    axis.set_title(f"{policy_name}: episode length")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{policy_name}_episode_length.png", dpi=180)
    plt.close(fig)
    return policy_name, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local SAC/PPO episode reward logs")
    parser.add_argument("--input-root", required=True, help="experiment directory containing pi*/episode_rewards.csv")
    parser.add_argument("--output-dir", default="", help="defaults to <input-root>/plots")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (input_root / "episode_rewards.csv").exists():
        # A single high-level PPO run stores its log directly in the run dir.
        policy_dirs = [input_root]
    else:
        # The low-level aggregate stores one log below each pi* directory.
        policy_dirs = sorted(path.parent for path in input_root.glob("pi*/episode_rewards.csv"))
    if not policy_dirs:
        raise FileNotFoundError(f"no policy episode logs found below {input_root}")

    plotted: list[tuple[str, list[dict[str, float]]]] = []
    for policy_dir in policy_dirs:
        plotted.append(_plot_policy(policy_dir, output_dir, args.window, plt))

    if len(plotted) > 1:
        fig, axis = plt.subplots(figsize=(9, 5))
        for policy_name, rows in plotted:
            steps = [row["global_step"] for row in rows]
            returns = [row["episode_return"] for row in rows]
            axis.plot(steps, _moving_average(returns, args.window), linewidth=1.6, label=policy_name)
        axis.set_xlabel("environment step")
        axis.set_ylabel("moving-average episode reward")
        axis.set_title("Low-level SAC training reward (diagnostic comparison)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "lowlevel_reward_comparison.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {len(plotted)} policy plot sets to {output_dir}")


if __name__ == "__main__":
    main()
