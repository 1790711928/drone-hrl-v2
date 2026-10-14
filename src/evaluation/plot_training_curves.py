"""Create paper-ready training-curve figures from locally collected CSV logs.

This utility is intentionally post-processing only.  It never loads a model,
creates an environment, or starts training.  The raw episode data remain in
``episode_rewards.csv``; generated PNG/CSV files are disposable plot views.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def _read_progress_rows(path: Path) -> list[dict[str, float]]:
    """Read SB3's step-indexed mean episode reward, when available."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(handle):
            step_text = row.get("time/total_timesteps", "")
            reward_text = row.get("rollout/ep_rew_mean", "")
            if not step_text or not reward_text:
                continue
            step = float(step_text)
            reward = float(reward_text)
            if math.isfinite(step) and math.isfinite(reward):
                rows.append({"global_step": step, "episode_return": reward})
        return rows


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


def _plot_policy(
    policy_dir: Path,
    output_dir: Path,
    window: int,
    source: str,
    max_step: int | None,
    plt,
) -> tuple[str, list[dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_path = policy_dir / "episode_rewards.csv"
    progress_path = policy_dir / "progress.csv"
    episode_rows = _read_rows(reward_path)
    progress_rows = _read_progress_rows(progress_path) if progress_path.exists() else []
    if source == "progress":
        rows = progress_rows or episode_rows
        using_progress = bool(progress_rows)
    else:
        rows = episode_rows or progress_rows
        using_progress = not bool(episode_rows) and bool(progress_rows)
    if max_step is not None:
        rows = [row for row in rows if row["global_step"] <= max_step]
        episode_rows = [row for row in episode_rows if row["global_step"] <= max_step]
    if not rows:
        raise ValueError(f"no data rows at or below --max-step={max_step}: {reward_path}")
    policy_name = policy_dir.name
    if using_progress:
        with (output_dir / f"{policy_name}_plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["global_step", "mean_episode_reward"])
            writer.writeheader()
            for row in progress_rows:
                writer.writerow({"global_step": row["global_step"], "mean_episode_reward": row["episode_return"]})
    else:
        _write_plot_data(output_dir / f"{policy_name}_plot_data.csv", rows, window)
    steps = [row["global_step"] for row in rows]
    returns = [row["episode_return"] for row in rows]
    moving = returns if using_progress else _moving_average(returns, window)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if using_progress:
        axis.plot(steps, returns, linewidth=1.8, label="average reward")
    else:
        axis.plot(steps, returns, alpha=0.28, linewidth=0.8, label="episode reward")
        axis.plot(steps, moving, linewidth=1.8, label="average reward")
    axis.set_xlabel("environment step")
    axis.set_ylabel("mean episode reward" if using_progress else "episode reward")
    axis.set_title(f"{policy_name}: training reward")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{policy_name}_reward_curve.png", dpi=180)
    plt.close(fig)

    if episode_rows:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(
            [row["global_step"] for row in episode_rows],
            [row["episode_length"] for row in episode_rows],
            linewidth=1.2,
        )
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
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="optional upper bound on displayed environment steps; raw CSV files remain unchanged",
    )
    parser.add_argument(
        "--source",
        choices=["episode", "progress"],
        default="episode",
        help="episode=raw episode rewards plus one average reward curve; progress=SB3 rollout mean",
    )
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.max_step is not None and args.max_step <= 0:
        parser.error("--max-step must be positive")

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
        plotted.append(_plot_policy(policy_dir, output_dir, args.window, args.source, args.max_step, plt))

    if len(plotted) > 1:
        fig, axis = plt.subplots(figsize=(9, 5))
        for policy_name, rows in plotted:
            steps = [row["global_step"] for row in rows]
            returns = [row["episode_return"] for row in rows]
            if args.source == "progress":
                axis.plot(steps, returns, linewidth=1.6, label=policy_name)
            else:
                axis.plot(steps, _moving_average(returns, args.window), linewidth=1.6, label=policy_name)
        axis.set_xlabel("environment step")
        axis.set_ylabel("average episode reward")
        axis.set_title("Low-level SAC training reward (diagnostic comparison)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "lowlevel_reward_comparison.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {len(plotted)} policy plot sets to {output_dir}")


if __name__ == "__main__":
    main()
