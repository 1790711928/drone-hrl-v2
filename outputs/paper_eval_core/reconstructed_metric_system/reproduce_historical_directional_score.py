"""Reproduce the historical directional behavior score from saved raw CSV data.

This script deliberately consumes the 2026-10-08 raw diagnostic export rather
than rerunning models. It writes a new score directory and does not generate a
heatmap; the matrix is reviewed before a final figure is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


POLICIES = ["pi1", "pi2", "pi3", "pi4"]
SCENARIO_ORDER = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]
METRIC_PAIRS = {
    "rear_close_threat": ("rear_gap_gain", "rear_gap_non_decreasing_ratio"),
    "flank_threat": ("side_gap_gain", "side_gap_non_decreasing_ratio"),
    "boundary_constrained": ("boundary_margin_gain", "safe_boundary_ratio"),
    "vertical_z_threat": ("vertical_away_displacement_gain", "distance_gain"),
}
SCENARIO_LABELS = {
    "rear_close_threat": "S1_rear_approach",
    "flank_threat": "S2_lateral_threat",
    "boundary_constrained": "S3_boundary_constraint",
    "vertical_z_threat": "S4_vertical_threat",
}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.5
    return (value - lower) / (upper - lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    refuse_nonempty(args.output_dir)

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("historical raw CSV is empty")

    required = {"scenario", "policy", "task_valid"}
    required.update(metric for pair in METRIC_PAIRS.values() for metric in pair)
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"historical CSV is missing required fields: {missing}")

    metric_names = sorted({metric for pair in METRIC_PAIRS.values() for metric in pair})
    bounds: dict[str, tuple[float, float]] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in raw_rows]
        bounds[metric] = (min(values), max(values))
        lower, upper = bounds[metric]
        for row in raw_rows:
            normalized = normalize(float(row[metric]), lower, upper)
            # The historical raw fields can be outside a nominal physical
            # range; this reconstruction uses their observed global range.
            row[f"{metric}_normalized"] = normalized

    scored_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        scenario = row["scenario"]
        if scenario not in METRIC_PAIRS:
            raise ValueError(f"unknown historical scenario: {scenario}")
        metric_1, metric_2 = METRIC_PAIRS[scenario]
        validity = float(row["task_valid"])
        score = validity * (
            float(row[f"{metric_1}_normalized"]) + float(row[f"{metric_2}_normalized"])
        ) / 2.0
        scored = dict(row)
        scored.update(
            {
                "indicator_1": metric_1,
                "indicator_2": metric_2,
                "V": validity,
                "behavior_score": score,
            }
        )
        scored_rows.append(scored)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored_rows:
        groups.setdefault((row["scenario"], row["policy"]), []).append(row)

    cell_rows: list[dict[str, Any]] = []
    matrix: dict[tuple[str, str], float] = {}
    for scenario in SCENARIO_ORDER:
        metric_1, metric_2 = METRIC_PAIRS[scenario]
        for policy in POLICIES:
            group = groups.get((scenario, policy), [])
            if not group:
                raise ValueError(f"missing historical cell: {scenario}/{policy}")
            value = mean([float(row["behavior_score"]) for row in group])
            matrix[(scenario, policy)] = value
            cell_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "policy": policy,
                    "indicator_1": metric_1,
                    "indicator_2": metric_2,
                    "episode_count": len(group),
                    "validity_rate": mean([float(row["V"]) for row in group]),
                    "indicator_1_raw_mean": mean([float(row[metric_1]) for row in group]),
                    "indicator_2_raw_mean": mean([float(row[metric_2]) for row in group]),
                    "indicator_1_normalized_mean": mean([float(row[f"{metric_1}_normalized"]) for row in group]),
                    "indicator_2_normalized_mean": mean([float(row[f"{metric_2}_normalized"]) for row in group]),
                    "behavior_score": value,
                }
            )

    score_path = args.output_dir / "historical_score_episode.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(scored_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored_rows)

    summary_path = args.output_dir / "historical_score_cell_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(cell_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    matrix_path = args.output_dir / "historical_score_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", *POLICIES])
        for scenario in SCENARIO_ORDER:
            writer.writerow([scenario, *[f"{matrix[(scenario, policy)]:.6f}" for policy in POLICIES]])

    normalization_path = args.output_dir / "historical_score_normalization.csv"
    with normalization_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "global_lower", "global_upper", "normalization_status"])
        for metric in metric_names:
            lower, upper = bounds[metric]
            writer.writerow([metric, lower, upper, "ASSUMPTION_GLOBAL_OBSERVED_RANGE"])

    report_lines = [
        "# Historical directional score reproduction",
        "",
        "Input: saved 2026-10-08 final_directional_diag behavior_metrics_raw.csv.",
        "This is a reconstruction from saved behavior data; it does not rerun models.",
        "",
        "## Explicit reconstruction choices",
        "",
        "- Indicator pairs are selected from the directional diagnostic fields.",
        "- Each metric uses one global observed min/max across all 80 raw rows.",
        "- Values are linearly mapped to [0, 1].",
        "- V is the saved task_valid field (1 for escaped, 0 otherwise).",
        "- Score = V * (normalized indicator 1 + normalized indicator 2) / 2.",
        "- No per-policy normalization is used.",
        "- No heatmap is generated by this script.",
        "",
        "## Diagonal check",
        "",
    ]
    all_diagonal = True
    for index, scenario in enumerate(SCENARIO_ORDER):
        target = POLICIES[index]
        values = {policy: matrix[(scenario, policy)] for policy in POLICIES}
        best = max(values, key=values.get)
        margin = values[target] - max(values[p] for p in POLICIES if p != target)
        all_diagonal = all_diagonal and best == target
        report_lines.append(
            f"- {scenario}: target={target} score={values[target]:.6f}; "
            f"best={best} score={values[best]:.6f}; margin={margin:.6f}"
        )
    report_lines.extend(["", f"all_four_diagonal_targets_best={all_diagonal}", ""])
    (args.output_dir / "historical_score_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "input_csv": str(args.input_csv.resolve()),
        "episode_rows": len(raw_rows),
        "cells": len(cell_rows),
        "metric_pairs": METRIC_PAIRS,
        "normalization": "global_observed_minmax",
        "validity_field": "task_valid",
        "heatmap_generated": False,
        "all_four_diagonal_targets_best": all_diagonal,
    }
    (args.output_dir / "historical_score_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved historical score matrix to {matrix_path}")
    print(f"Saved report to {args.output_dir / 'historical_score_report.md'}")
    print(f"all_four_diagonal_targets_best={all_diagonal}")


if __name__ == "__main__":
    main()
