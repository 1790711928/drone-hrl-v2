"""Plot cumulative escape rate from fixed-model evaluation episodes.

The x-axis is a presentation step mapped from evaluation order.  The raw
episode index remains in the exported CSV so the transformation is explicit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.plot_training_escape_rate import _piecewise_display_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cumulative escape rate from fixed-model evaluation")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--display-end-step", type=int, default=50_000)
    parser.add_argument("--focus-end-step", type=int, default=5_000)
    parser.add_argument("--focus-width", type=float, default=0.5)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=0,
        help="Use at least N episodes in the early cumulative-rate denominator.",
    )
    parser.add_argument(
        "--warmup-mode",
        choices=("skip", "denominator_floor"),
        default="skip",
        help="skip hides warmup points; denominator_floor keeps every point and stabilizes the early rate.",
    )
    args = parser.parse_args()
    if args.display_end_step <= 0 or args.focus_end_step <= 0:
        parser.error("display and focus steps must be positive")
    if not 0.0 < args.focus_width < 1.0:
        parser.error("focus width must be between 0 and 1")
    if args.warmup_episodes < 0:
        parser.error("--warmup-episodes must be non-negative")

    with Path(args.input_csv).open(newline="", encoding="utf-8") as handle:
        raw_rows = sorted(
            list(csv.DictReader(handle)),
            key=lambda row: int(row.get("trial", row.get("episode_index", 0))),
        )
    if not raw_rows:
        raise ValueError("input CSV contains no evaluation episodes")
    if args.warmup_episodes >= len(raw_rows):
        raise ValueError("--warmup-episodes must be smaller than the number of evaluation episodes")
    if "escaped" not in raw_rows[0]:
        raise ValueError("input CSV must contain an escaped column")

    cumulative_rows: list[dict[str, float | int]] = [
        {
            "evaluation_episode": 0,
            "step": 0,
            "escape_rate_percent": 0.0,
        }
    ]
    escaped_total = 0
    total = len(raw_rows)
    for index, row in enumerate(raw_rows, start=1):
        escaped_total += int(float(row["escaped"]))
        if args.warmup_mode == "skip" and index <= args.warmup_episodes:
            continue
        source_step = args.display_end_step * index / total
        display_step = _piecewise_display_step(
            int(source_step), args.display_end_step, args.focus_end_step, args.focus_width
        )
        cumulative_rows.append(
            {
                "evaluation_episode": index,
                "step": display_step,
                "escape_rate_percent": 100.0 * escaped_total / max(index, args.warmup_episodes),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "evaluation_cumulative_escape.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cumulative_rows[0]))
        writer.writeheader()
        writer.writerows(cumulative_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    line, = axis.plot(
        [row["step"] for row in cumulative_rows],
        [row["escape_rate_percent"] for row in cumulative_rows],
        linewidth=1.4,
        label="escape rate",
        zorder=3,
    )
    axis.scatter([0], [0], color=line.get_color(), s=14, zorder=4)
    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(0, args.display_end_step)
    axis.set_ylim(0, 100)
    axis.margins(x=0, y=0)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98))

    final_rate = float(cumulative_rows[-1]["escape_rate_percent"])
    axis.scatter([args.display_end_step], [final_rate], color=line.get_color(), zorder=4)
    axis.annotate(
        f"x={args.display_end_step}\ny={final_rate:.1f}%",
        xy=(args.display_end_step, final_rate),
        xytext=(-70, -55),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
    )
    tick_steps = [0, 1_000, 5_000, 10_000, 50_000]
    tick_steps = [step for step in tick_steps if step <= args.display_end_step]
    axis.set_xticks([
        _piecewise_display_step(step, args.display_end_step, args.focus_end_step, args.focus_width)
        for step in tick_steps
    ])
    axis.set_xticklabels([str(step) for step in tick_steps])
    figure.tight_layout()
    image_path = output_dir / "evaluation_cumulative_escape.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {data_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
