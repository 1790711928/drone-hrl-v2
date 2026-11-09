"""Summarize reward changes over training bins without running an environment."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _read_episode_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "global_step": float(row["global_step"]),
                    "episode_return": float(row["episode_return"]),
                }
            )
        return rows


def _summarize(rows: list[dict[str, float]], bin_steps: int) -> list[dict[str, float]]:
    bins: dict[int, list[float]] = {}
    for row in rows:
        bin_start = int(row["global_step"] // bin_steps) * bin_steps
        bins.setdefault(bin_start, []).append(row["episode_return"])
    summary: list[dict[str, float]] = []
    for bin_start in sorted(bins):
        values = bins[bin_start]
        summary.append(
            {
                "bin_start_step": bin_start,
                "bin_end_step": bin_start + bin_steps,
                "episode_count": len(values),
                "mean_episode_return": statistics.mean(values),
                "median_episode_return": statistics.median(values),
                "std_episode_return": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min_episode_return": min(values),
                "max_episode_return": max(values),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one policy's raw episode returns by training-step bin")
    parser.add_argument("--input-root", required=True, help="low-level aggregate or single run directory")
    parser.add_argument("--policy-dir", default="pi2_flank", help="policy directory below input-root")
    parser.add_argument("--bin-steps", type=int, default=5_000)
    parser.add_argument("--output", default="", help="defaults to <policy-dir>/reward_bins.csv")
    args = parser.parse_args()
    if args.bin_steps <= 0:
        parser.error("--bin-steps must be positive")

    input_root = Path(args.input_root)
    policy_dir = input_root / args.policy_dir if (input_root / args.policy_dir).is_dir() else input_root
    source = policy_dir / "episode_rewards.csv"
    rows = _read_episode_rows(source)
    if not rows:
        raise ValueError(f"no episode reward rows found: {source}")
    summary = _summarize(rows, args.bin_steps)
    output = Path(args.output) if args.output else policy_dir / "reward_bins.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} reward bins to {output}")
    for row in summary:
        print(
            f"{int(row['bin_start_step']):>6}-{int(row['bin_end_step']):<6} "
            f"n={int(row['episode_count']):>4} "
            f"mean={row['mean_episode_return']:>8.2f} "
            f"std={row['std_episode_return']:>7.2f}"
        )


if __name__ == "__main__":
    main()
