"""Overlay high-level win-rate curves from multiple evaluation CSV files.

Each input CSV must contain ``checkpoint_step`` and ``win_rate`` columns, as
written by :mod:`evaluate_highlevel_checkpoints`.  The figure is intentionally
a line plot: the x-axis is high-level training step and each series is a
system/ablation, so no bar-chart aggregation is introduced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_series(spec: str) -> tuple[str, list[int], list[float]]:
    """Read ``LABEL=CSV_PATH`` and return steps plus win rates in percent."""
    if "=" not in spec:
        raise ValueError(f"series must use LABEL=CSV_PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise ValueError(f"series label is empty: {spec!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"series CSV is empty: {path}")
    try:
        steps = [int(row["checkpoint_step"]) for row in rows]
        rates = [100.0 * float(row["win_rate"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"series CSV must contain numeric checkpoint_step/win_rate columns: {path}") from exc
    order = sorted(range(len(steps)), key=steps.__getitem__)
    return label, [steps[index] for index in order], [rates[index] for index in order]


def plot_win_rate_curves(series_specs: list[str], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for spec in series_specs:
        label, steps, rates = _read_series(spec)
        axis.plot(steps, rates, marker="o", linewidth=1.8, markersize=4, label=label)
    axis.set_xlabel("training step")
    axis.set_ylabel("win rate (%)")
    axis.set_title(title)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay high-level win-rate curves")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="one evaluation CSV per series; repeat for complete system and ablations",
    )
    parser.add_argument("--output", required=True, help="PNG path")
    parser.add_argument("--title", default="High-level win-rate comparison")
    args = parser.parse_args()
    plot_win_rate_curves(args.series, Path(args.output), args.title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
