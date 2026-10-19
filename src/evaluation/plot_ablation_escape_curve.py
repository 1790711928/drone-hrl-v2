"""Plot an ablation checkpoint escape-rate curve with the frozen paper style."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.plot_training_escape_rate import _piecewise_display_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation escape rate by PPO checkpoint")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--display-end-step", type=int, default=50_000)
    parser.add_argument("--focus-end-step", type=int, default=5_000)
    parser.add_argument("--focus-width", type=float, default=0.5)
    args = parser.parse_args()
    if args.display_end_step <= 0 or args.focus_end_step <= 0:
        parser.error("display and focus steps must be positive")
    if not 0.0 < args.focus_width < 1.0:
        parser.error("focus width must be between 0 and 1")

    with Path(args.input_csv).open(newline="", encoding="utf-8") as handle:
        rows = sorted(
            [
                (int(row["checkpoint_step"]), 100.0 * float(row["win_rate"]))
                for row in csv.DictReader(handle)
                if int(row["checkpoint_step"]) <= args.display_end_step
            ]
        )
    if not rows:
        raise ValueError("input CSV contains no checkpoint rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_rows = [(0, 0.0), *rows]
    data_path = output_dir / "ablation_escape_rate_curve.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "escape_rate_percent"])
        writer.writeheader()
        writer.writerows({"step": step, "escape_rate_percent": rate} for step, rate in display_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x_values = [
        _piecewise_display_step(step, args.display_end_step, args.focus_end_step, args.focus_width)
        for step, _ in display_rows
    ]
    y_values = [rate for _, rate in display_rows]
    line, = axis.plot(x_values, y_values, linewidth=1.6, label="escape rate", zorder=3)
    axis.scatter([0], [0], color=line.get_color(), s=14, zorder=4)
    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(0, args.display_end_step)
    axis.set_ylim(0, 100)
    axis.margins(x=0, y=0)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98))

    final_step, final_rate = rows[-1]
    final_x = _piecewise_display_step(final_step, args.display_end_step, args.focus_end_step, args.focus_width)
    axis.scatter([final_x], [final_rate], color=line.get_color(), zorder=4)
    axis.annotate(
        f"x={final_step}\ny={final_rate:.1f}%",
        xy=(final_x, final_rate),
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
    image_path = output_dir / "ablation_escape_rate_curve.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {data_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
