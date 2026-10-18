"""Plot the dense training-time escape-rate diagnostic.

The input is the training callback's ``episode_rewards.csv`` after the
outcome fields have been enabled.  The plotted value is a rolling mean over
completed episodes, indexed by the actual global training step.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path


def _read_escape_rows(path: Path, window: int, mode: str = "rolling") -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "escaped" not in fields or "global_step" not in fields:
            raise ValueError(
                f"{path} does not contain outcome logging. Re-run training with the updated EpisodeRewardCallback."
            )
        history: deque[int] = deque(maxlen=window)
        output: list[dict[str, float | int]] = []
        total_episodes = 0
        total_escaped = 0
        for row in reader:
            escaped = int(float(row["escaped"]))
            total_episodes += 1
            total_escaped += escaped
            history.append(escaped)
            escape_rate = (
                total_escaped / total_episodes
                if mode == "cumulative"
                else sum(history) / len(history)
            )
            output.append(
                {
                    "global_step": int(float(row["global_step"])),
                    "episode_index": int(float(row.get("episode_index", len(output) + 1))),
                    "window_episodes": total_episodes if mode == "cumulative" else len(history),
                    "escape_rate": escape_rate,
                }
            )
    if mode == "cumulative":
        output.insert(
            0,
            {
                "global_step": 0,
                "episode_index": 0,
                "window_episodes": 0,
                "escape_rate": 0.0,
            },
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rolling training escape rate")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--mode", choices=("rolling", "cumulative"), default="rolling")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")

    rows = _read_escape_rows(Path(args.input_csv), args.window, args.mode)
    if not rows:
        raise ValueError("input CSV contains no episode rows")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "training_escape_rate.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["global_step"] for row in rows],
        [100.0 * float(row["escape_rate"]) for row in rows],
        linewidth=1.4,
        label="escape rate",
    )
    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    image_path = output_dir / "training_escape_rate.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {data_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
