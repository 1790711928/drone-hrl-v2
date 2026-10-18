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


def _piecewise_display_step(
    step: int,
    total_step: int,
    focus_end: int | None = None,
    focus_width: float = 1.0 / 3.0,
) -> float:
    """Map actual steps to a three-segment display axis."""
    total_step = max(int(total_step), 1)
    if focus_end is not None:
        focus_end = min(max(int(focus_end), 1), total_step)
        first_width = min(max(float(focus_width), 0.01), 0.99)
        if step <= focus_end:
            return total_step * first_width * step / focus_end
        tail_fraction = (step - focus_end) / max(total_step - focus_end, 1)
        return total_step * (first_width + (1.0 - first_width) * tail_fraction)

    first_end = min(10_000, total_step)
    second_end = min(20_000, total_step)
    weights = (1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0)
    weight_sum = sum(weights)
    widths = tuple(weight / weight_sum for weight in weights)
    if step <= first_end:
        fraction = step / max(first_end, 1)
        return total_step * widths[0] * fraction
    if step <= second_end:
        fraction = (step - first_end) / max(second_end - first_end, 1)
        return total_step * (widths[0] + widths[1] * fraction)
    fraction = (step - second_end) / max(total_step - second_end, 1)
    return total_step * (widths[0] + widths[1] + widths[2] * fraction)


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
    parser.add_argument(
        "--focus-steps",
        type=int,
        default=None,
        help="Show the first N steps in the main axes and the full run in an inset.",
    )
    parser.add_argument(
        "--piecewise-axis",
        action="store_true",
        help="Use the three-segment visual x-axis: 0-10k, 10k-20k, 20k-end.",
    )
    parser.add_argument(
        "--piecewise-focus-end",
        type=int,
        default=None,
        help="Use a two-segment piecewise axis with the first segment ending at this step.",
    )
    parser.add_argument(
        "--piecewise-focus-width",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of display width assigned to the focused first segment.",
    )
    parser.add_argument(
        "--display-end-step",
        type=int,
        default=None,
        help="Clamp the display axis and final annotation to this actual step.",
    )
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
    focus_rows = rows
    if args.focus_steps is not None:
        if args.focus_steps <= 0:
            parser.error("--focus-steps must be positive")
        focus_rows = [row for row in rows if int(row["global_step"]) <= args.focus_steps]
        if len(focus_rows) < 2:
            raise ValueError("--focus-steps must include at least two recorded points")
    if args.focus_steps is not None and args.piecewise_axis:
        parser.error("--focus-steps and --piecewise-axis cannot be combined")
    if args.piecewise_focus_end is not None and not args.piecewise_axis:
        parser.error("--piecewise-focus-end requires --piecewise-axis")
    total_step = max(int(row["global_step"]) for row in rows)
    if args.display_end_step is not None:
        if args.display_end_step <= 0:
            parser.error("--display-end-step must be positive")
        total_step = min(total_step, args.display_end_step)
    if args.piecewise_focus_width <= 0 or args.piecewise_focus_width >= 1:
        parser.error("--piecewise-focus-width must be between 0 and 1")

    def x_value(row: dict[str, float | int]) -> float:
        step = min(int(row["global_step"]), total_step)
        return (
            _piecewise_display_step(step, total_step, args.piecewise_focus_end, args.piecewise_focus_width)
            if args.piecewise_axis
            else float(step)
        )

    line, = axis.plot(
        [x_value(row) for row in focus_rows],
        [100.0 * float(row["escape_rate"]) for row in focus_rows],
        linewidth=1.4,
        label="escape rate",
        zorder=3,
    )
    axis.scatter([0], [0], color=line.get_color(), s=14, zorder=4)
    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(left=0)
    if args.focus_steps is not None:
        axis.set_xlim(0, args.focus_steps)
    elif args.piecewise_axis:
        axis.set_xlim(0, total_step)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98))

    final_row = rows[-1]
    final_step = args.display_end_step if args.display_end_step is not None else int(final_row["global_step"])
    final_rate = 100.0 * float(final_row["escape_rate"])
    final_display_step = x_value(final_row)
    if args.focus_steps is None:
        axis.scatter([final_display_step], [final_rate], zorder=3)
        axis.annotate(
            f"x={final_step}\ny={final_rate:.1f}%",
            xy=(final_display_step, final_rate),
            xytext=(-70, -55),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
        )
    else:
        inset = axis.inset_axes([0.53, 0.12, 0.44, 0.36])
        inset.plot(
            [row["global_step"] for row in rows],
            [100.0 * float(row["escape_rate"]) for row in rows],
            linewidth=1.0,
        )
        inset.scatter([final_step], [final_rate], s=18, zorder=3)
        inset.annotate(
            f"x={final_step}\ny={final_rate:.1f}%",
            xy=(final_step, final_rate),
            xytext=(-55, 10),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.85},
        )
        inset.set_xlim(0, max(final_step, 1))
        inset.set_ylim(0, 100)
        inset.grid(alpha=0.2)
        inset.tick_params(labelsize=7)
    if args.piecewise_axis:
        if args.piecewise_focus_end is not None:
            tick_steps = [step for step in (0, 1_000, 2_000, args.piecewise_focus_end, 10_000, 20_000, 30_000, 40_000, 50_000) if step <= total_step]
            boundary_display = _piecewise_display_step(
                args.piecewise_focus_end,
                total_step,
                args.piecewise_focus_end,
                args.piecewise_focus_width,
            )
        else:
            tick_steps = [step for step in (0, 5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000) if step <= total_step]
        if args.piecewise_focus_end is not None:
            tick_steps = [step for step in (0, 1_000, 5_000, 10_000, 50_000) if step <= total_step]
        axis.set_xticks([
            _piecewise_display_step(step, total_step, args.piecewise_focus_end, args.piecewise_focus_width)
            for step in tick_steps
        ])
        axis.set_xticklabels([str(step) for step in tick_steps])
    axis.margins(x=0, y=0)
    axis.set_xlim(0, total_step)
    axis.set_ylim(0, 100)
    figure.tight_layout()
    image_path = output_dir / "training_escape_rate.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {data_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
