"""Create the final annotated heatmap from a reviewed historical score matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.matrix_csv.is_file():
        raise FileNotFoundError(args.matrix_csv)

    with args.matrix_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"scenario", "pi1", "pi2", "pi3", "pi4"}:
        raise ValueError("matrix CSV must contain scenario, pi1, pi2, pi3, pi4")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError("matplotlib and numpy are required for heatmap generation") from exc

    labels = ["S1 rear", "S2 lateral", "S3 boundary", "S4 vertical"]
    policies = ["π1", "π2", "π3", "π4"]
    data = np.array([[float(row[f"pi{i}"]) for i in range(1, 5)] for row in rows], dtype=float)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    image = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), labels=policies, fontsize=11)
    ax.set_yticks(range(4), labels=labels, fontsize=11)
    ax.set_xlabel("Low-level policy", fontsize=12)
    ax.set_ylabel("Threat scenario", fontsize=12)
    ax.set_title("Historical directional behavior score", fontsize=13, pad=12)
    for row_index in range(4):
        for column_index in range(4):
            value = float(data[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.52 else "black",
                fontsize=11,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Behavior score", fontsize=11)
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {args.output_png}")


if __name__ == "__main__":
    main()
