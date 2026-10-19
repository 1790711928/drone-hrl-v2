"""Plot the four frozen escape-rate curves in one paper-style figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.plot_training_escape_rate import _piecewise_display_step


METHODS = (
    ("hrl", "HRL", "#1f77b4"),
    ("random_disable", "随机禁用策略", "#ff7f0e"),
    ("rule_guided", "规则引导系统", "#2ca02c"),
    ("single_sac", "单一 SAC 策略", "#d62728"),
)


def _read_curve(path: Path, display_end_step: int, focus_end_step: int, focus_width: float) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"curve CSV is empty: {path}")
    fields = set(rows[0])
    if {"global_step", "escape_rate"}.issubset(fields):
        points = [
            (
                float(row["global_step"]),
                100.0 * float(row["escape_rate"]),
            )
            for row in rows
        ]
        mapped = [
            {
                "step": _piecewise_display_step(
                    int(min(max(step, 0.0), display_end_step)),
                    display_end_step,
                    focus_end_step,
                    focus_width,
                ),
                "escape_rate_percent": rate,
            }
            for step, rate in points
        ]
    elif {"step", "escape_rate_percent"}.issubset(fields):
        mapped = [
            {
                "step": float(row["step"]),
                "escape_rate_percent": float(row["escape_rate_percent"]),
            }
            for row in rows
        ]
    else:
        raise ValueError(
            f"unsupported curve CSV schema: {path}; expected global_step/escape_rate or step/escape_rate_percent"
        )
    mapped.sort(key=lambda row: row["step"])
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the four frozen escape-rate curves")
    parser.add_argument("--hrl-csv", required=True)
    parser.add_argument("--random-disable-csv", required=True)
    parser.add_argument("--rule-guided-csv", required=True)
    parser.add_argument("--single-sac-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--display-end-step", type=int, default=50_000)
    parser.add_argument("--focus-end-step", type=int, default=5_000)
    parser.add_argument("--focus-width", type=float, default=0.5)
    args = parser.parse_args()
    if args.display_end_step <= 0 or args.focus_end_step <= 0:
        parser.error("display and focus steps must be positive")
    if not 0.0 < args.focus_width < 1.0:
        parser.error("focus width must be between 0 and 1")

    input_paths = {
        "hrl": Path(args.hrl_csv),
        "random_disable": Path(args.random_disable_csv),
        "rule_guided": Path(args.rule_guided_csv),
        "single_sac": Path(args.single_sac_csv),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing curve CSV(s): {missing}")

    curves = {
        key: _read_curve(path, args.display_end_step, args.focus_end_step, args.focus_width)
        for key, path in input_paths.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "combined_escape_rates.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "step", "escape_rate_percent"])
        writer.writeheader()
        for key, _, _ in METHODS:
            for row in curves[key]:
                writer.writerow(
                    {
                        "method": key,
                        "step": row["step"],
                        "escape_rate_percent": row["escape_rate_percent"],
                    }
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    final_rates: dict[str, float] = {}
    for key, label, color in METHODS:
        rows = curves[key]
        x_values = [row["step"] for row in rows]
        y_values = [row["escape_rate_percent"] for row in rows]
        axis.plot(x_values, y_values, linewidth=1.4, color=color, label=label, zorder=2)
        final_rates[key] = float(y_values[-1])
        axis.scatter([x_values[-1]], [y_values[-1]], color=color, s=28, zorder=4)

    axis.set_xlabel("step")
    axis.set_ylabel("escape rate (%)")
    axis.set_xlim(0, args.display_end_step)
    axis.set_ylim(0, 100)
    axis.margins(x=0, y=0)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", frameon=True)

    # Put method/value boxes in a right-side column so endpoint values remain readable.
    figure.subplots_adjust(right=0.78)
    box_y = {"hrl": 0.82, "random_disable": 0.66, "rule_guided": 0.50, "single_sac": 0.34}
    for key, label, color in METHODS:
        axis.text(
            1.02,
            box_y[key],
            f"{label}\ny={final_rates[key]:.1f}%",
            transform=axis.transAxes,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": color, "lw": 1.6},
            color="#222222",
        )

    tick_steps = [step for step in (0, 1_000, 5_000, 10_000, 50_000) if step <= args.display_end_step]
    axis.set_xticks([
        _piecewise_display_step(step, args.display_end_step, args.focus_end_step, args.focus_width)
        for step in tick_steps
    ])
    axis.set_xticklabels([str(step) for step in tick_steps])
    figure.tight_layout()
    image_path = output_dir / "combined_escape_rates.png"
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {combined_path}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()
