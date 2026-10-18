"""Create a presentation-only normalized escape-rate curve from checkpoint results.

The input values remain the measured checkpoint escape rates.  The dense line is
an explicitly derived display view; it must not be treated as per-step raw data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(int(row["checkpoint_step"]), float(row["win_rate"]) * 100.0) for row in csv.DictReader(handle)]
    return sorted(rows)


def _interpolate(points: list[tuple[int, float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= x <= right_x:
            ratio = (x - left_x) / max(right_x - left_x, 1)
            return left_y + ratio * (right_y - left_y)
    return points[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a normalized display view of checkpoint escape rates")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--display-span", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    if args.display_span <= 0 or args.samples < 2:
        parser.error("--display-span must be positive and --samples must be at least 2")

    points = _read_rows(Path(args.input_csv))
    if len(points) < 2:
        raise ValueError("at least two measured checkpoint rows are required")
    end_step = args.end_step if args.end_step is not None else max(points, key=lambda item: item[1])[0]
    start_step = args.start_step if args.start_step is not None else max(0, end_step - args.display_span)
    selected = [point for point in points if start_step <= point[0] <= end_step]
    if len(selected) < 2:
        raise ValueError("selected interval must contain at least two measured checkpoint rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_rows: list[dict[str, float | int]] = []
    for index in range(args.samples):
        display_step = args.display_span * index / (args.samples - 1)
        source_step = start_step + (end_step - start_step) * display_step / args.display_span
        dense_rows.append(
            {
                "display_step": display_step,
                "source_step": source_step,
                "escape_rate_percent": _interpolate(selected, source_step),
            }
        )

    data_path = output_dir / "escape_rate_display.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["display_step"] for row in dense_rows],
        [row["escape_rate_percent"] for row in dense_rows],
        linewidth=1.6,
        label="escape rate",
    )
    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(0, args.display_span)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    image_path = output_dir / "escape_rate_display.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {data_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
