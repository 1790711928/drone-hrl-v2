"""Plot a contrast-enhanced relative score without hard 0/100 endpoints.

The raw score matrix is unchanged.  For each scenario row, a softmax turns
the four scores into positive relative weights that sum to one.  The value
shown in each cell is the relative score (decimal form, not a percentage).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICIES = ["pi1", "pi2", "pi3", "pi4"]
SCENARIO_LABELS = {
    "rear_close_threat": "S1 rear",
    "flank_threat": "S2 lateral",
    "boundary_constrained": "S3 boundary",
    "vertical_z_threat": "S4 vertical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beta <= 0:
        raise ValueError("beta must be positive")
    if args.scale <= 0:
        raise ValueError("scale must be positive")
    if not args.matrix_csv.is_file():
        raise FileNotFoundError(args.matrix_csv)

    with args.matrix_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"scenario", *POLICIES}
    if not rows or set(rows[0]) != expected:
        raise ValueError("matrix CSV must contain scenario, pi1, pi2, pi3, pi4")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError("matplotlib and numpy are required") from exc

    raw = np.array([[float(row[policy]) for policy in POLICIES] for row in rows], dtype=float)
    relative = np.zeros_like(raw)
    for row_index in range(raw.shape[0]):
        shifted = args.beta * (raw[row_index] - raw[row_index].max())
        weights = np.exp(shifted)
        relative[row_index] = weights / weights.sum()
    display = relative * args.scale

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "policy", "raw_score", "relative_score", "beta", "scale"])
        for row_index, row in enumerate(rows):
            for column_index, policy in enumerate(POLICIES):
                writer.writerow(
                    [
                        row["scenario"],
                        policy,
                        f"{raw[row_index, column_index]:.6f}",
                        f"{display[row_index, column_index]:.6f}",
                        f"{args.beta:.3f}",
                        f"{args.scale:.3f}",
                    ]
                )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    image = ax.imshow(display, cmap="YlGnBu", vmin=0.0, vmax=args.scale, aspect="auto")
    ax.set_xticks(range(4), labels=["π1", "π2", "π3", "π4"], fontsize=12)
    ax.set_yticks(
        range(len(rows)),
        labels=[SCENARIO_LABELS.get(row["scenario"], row["scenario"]) for row in rows],
        fontsize=12,
    )
    ax.set_xlabel("Low-level policy", fontsize=13)
    ax.set_ylabel("Threat scenario", fontsize=13)

    for row_index in range(len(rows)):
        for column_index in range(len(POLICIES)):
            ax.text(
                column_index,
                row_index,
                f"{display[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Matching score", fontsize=11)
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contrast matrix to {args.output_csv}")
    print(f"Saved heatmap to {args.output_png}")


if __name__ == "__main__":
    main()
