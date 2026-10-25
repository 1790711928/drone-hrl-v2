"""Evaluate the Chapter 3 two-indicator low-level behavior score matrix.

The paper defines two behavior indicators for each skill and combines their
normalized values as ``S = V * (m1_normalized + m2_normalized) / 2``.  This
module evaluates all four frozen SAC policies in all four canonical threat
scenarios using paired initial states, computes that score per episode, and
exports the 4x4 score matrix.  Terminal escape rate is retained as an
auxiliary outcome, never used as the heatmap value.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.eval_lowlevel_behavior_raw import (
    MODEL_FILENAMES,
    POLICIES,
    SCENARIOS,
    run_paired_episode,
    sample_paired_initial_state,
)


METRIC_PAIRS = {
    "rear_close_threat": ("m_turn", "g_v"),
    "flank_threat": ("side_gap_gain", "side_gap_non_decreasing_ratio"),
    "boundary_constrained": ("boundary_margin_gain", "safe_boundary_ratio"),
    "vertical_z_threat": ("vertical_separation_gain", "distance_gain"),
}


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.policy_checkpoints is not None:
        return [Path(value) for value in args.policy_checkpoints]
    checkpoint_dir = Path(args.checkpoint_dir)
    return [
        Path(value) if value else checkpoint_dir / filename
        for value, filename in zip(
            [args.policy1_checkpoint, args.policy2_checkpoint, args.policy3_checkpoint, args.policy4_checkpoint],
            MODEL_FILENAMES,
        )
    ]


def load_models(paths: list[Path], device: str) -> list[Any]:
    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required to load SAC checkpoints.") from exc
    return [SAC.load(str(path), device=device) for path in paths]


def minmax_normalize(rows: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows]
    lower = min(values)
    upper = max(values)
    span = upper - lower
    for row in rows:
        value = float(row[metric])
        row[f"{metric}_normalized"] = 0.5 if span <= 1e-12 else (value - lower) / span
    return lower, upper


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["scenario", "policy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", *POLICIES])
        for scenario in SCENARIOS:
            writer.writerow([scenario] + [f"{matrix[(scenario, policy)]:.6f}" for policy in POLICIES])


def save_heatmap(matrix: dict[tuple[str, str], float], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[heatmap] matplotlib not installed; skipped {path}")
        return

    data = np.array([[matrix[(scenario, policy)] for policy in POLICIES] for scenario in SCENARIOS])
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    image = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4), labels=POLICIES)
    ax.set_yticks(range(4), labels=["rear", "flank", "boundary", "vertical"])
    ax.set_xlabel("Low-level policy")
    ax.set_ylabel("Threat scenario")
    ax.set_title("Low-level behavior adaptation score matrix")
    for row_index in range(4):
        for column_index in range(4):
            value = float(data[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Behavior adaptation score S")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[heatmap] saved: {path}")


def write_summary(path: Path, matrix: dict[tuple[str, str], float], episodes: int) -> None:
    lines = [
        "# Chapter 3 low-level behavior adaptation score",
        "",
        "The heatmap value is S = V * (normalized indicator 1 + normalized indicator 2) / 2.",
        "V is 1 only for an escaped episode and 0 for captured, out-of-bounds, or timeout episodes.",
        f"Episodes per policy/scenario cell: {episodes}",
        "",
        "## Diagonal check",
        "",
    ]
    all_diagonal = True
    for index, scenario in enumerate(SCENARIOS):
        owner = POLICIES[index]
        values = {policy: matrix[(scenario, policy)] for policy in POLICIES}
        best_policy = max(values, key=values.get)
        margin = values[owner] - max(value for policy, value in values.items() if policy != owner)
        all_diagonal = all_diagonal and best_policy == owner
        lines.append(
            f"- {scenario}: diagonal={owner} ({values[owner]:.4f}), "
            f"best={best_policy} ({values[best_policy]:.4f}), margin={margin:.4f}"
        )
    lines.extend(["", f"Diagonal highest in all four scenarios: {all_diagonal}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 3 low-level two-indicator behavior score matrix")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--policy-checkpoints", nargs=4, default=None, metavar=("PI1", "PI2", "PI3", "PI4"))
    parser.add_argument("--policy1-checkpoint", default="")
    parser.add_argument("--policy2-checkpoint", default="")
    parser.add_argument("--policy3-checkpoint", default="")
    parser.add_argument("--policy4-checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="outputs/evaluation/lowlevel_behavior_score_matrix")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_paths = resolve_model_paths(args)
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
    models = load_models(model_paths, args.device)

    raw_rows: list[dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial_id in range(args.episodes):
            trial_seed = args.seed + scenario_index * 100_000 + trial_id
            initial_state = sample_paired_initial_state(scenario, trial_seed)
            for policy, model in zip(POLICIES, models):
                raw_rows.append(
                    run_paired_episode(
                        model,
                        scenario=scenario,
                        policy=policy,
                        trial_id=trial_id,
                        seed=trial_seed,
                        initial_state=initial_state,
                        boundary_safe_threshold=0.35,
                        vertical_safe_threshold=0.20,
                    )
                )

    metric_names = sorted({metric for pair in METRIC_PAIRS.values() for metric in pair})
    normalization_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        lower, upper = minmax_normalize(raw_rows, metric)
        normalization_rows.append({"metric": metric, "min": lower, "max": upper})

    for row in raw_rows:
        metric_1, metric_2 = METRIC_PAIRS[str(row["scenario"])]
        row["score_indicator_1"] = metric_1
        row["score_indicator_2"] = metric_2
        row["validity_gate"] = int(row["task_valid"])
        row["behavior_score"] = row["validity_gate"] * 0.5 * (
            float(row[f"{metric_1}_normalized"]) + float(row[f"{metric_2}_normalized"])
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["scenario"]), str(row["policy"]))].append(row)

    score_matrix: dict[tuple[str, str], float] = {}
    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        metric_1, metric_2 = METRIC_PAIRS[scenario]
        for policy in POLICIES:
            group = grouped[(scenario, policy)]
            score_matrix[(scenario, policy)] = float(np.mean([row["behavior_score"] for row in group]))
            summary_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "indicator_1": metric_1,
                    "indicator_2": metric_2,
                    "indicator_1_raw_mean": float(np.mean([row[metric_1] for row in group])),
                    "indicator_2_raw_mean": float(np.mean([row[metric_2] for row in group])),
                    "indicator_1_normalized_mean": float(np.mean([row[f"{metric_1}_normalized"] for row in group])),
                    "indicator_2_normalized_mean": float(np.mean([row[f"{metric_2}_normalized"] for row in group])),
                    "behavior_score": score_matrix[(scenario, policy)],
                    "validity_rate": float(np.mean([row["validity_gate"] for row in group])),
                    "escape_rate_auxiliary": float(np.mean([row["success"] for row in group])),
                    "capture_rate_auxiliary": float(np.mean([row["captured"] for row in group])),
                    "out_of_bounds_rate_auxiliary": float(np.mean([row["out_of_bounds"] for row in group])),
                    "timeout_rate_auxiliary": float(np.mean([row["timeout"] for row in group])),
                }
            )

    write_rows(out_dir / "behavior_score_episode_raw.csv", raw_rows)
    write_rows(out_dir / "behavior_score_cell_summary.csv", summary_rows)
    write_rows(out_dir / "behavior_score_normalization.csv", normalization_rows)
    write_matrix(out_dir / "behavior_score_matrix.csv", score_matrix)
    save_heatmap(score_matrix, out_dir / "behavior_score_heatmap.png")
    write_summary(out_dir / "behavior_score_summary.md", score_matrix, args.episodes)

    print(f"[csv] saved: {out_dir / 'behavior_score_matrix.csv'}")
    print(f"[csv] saved: {out_dir / 'behavior_score_cell_summary.csv'}")
    print(f"[summary] saved: {out_dir / 'behavior_score_summary.md'}")
    for scenario in SCENARIOS:
        values = ", ".join(f"{policy}={score_matrix[(scenario, policy)]:.3f}" for policy in POLICIES)
        print(f"{scenario}: {values}")


if __name__ == "__main__":
    main()
