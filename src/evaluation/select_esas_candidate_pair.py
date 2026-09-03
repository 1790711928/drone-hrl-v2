from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation import eval_lowlevel_esas as esas_eval

POLICIES = esas_eval.POLICIES
SCENARIOS = esas_eval.SCENARIOS
DIAGONAL_POLICY = esas_eval.DIAGONAL_POLICY


def find_candidates(candidate_dir: Path) -> tuple[list[Path], list[Path]]:
    pi1_dir = candidate_dir / "pi1_rear"
    pi4_dir = candidate_dir / "pi4_vertical"
    return sorted(pi1_dir.glob("*.zip")), sorted(pi4_dir.glob("*.zip"))


def second_best(values: dict[str, float], owner: str) -> float:
    return max(value for policy, value in values.items() if policy != owner)


def matrix_values(rows: list[dict[str, str | float]], scenario: str) -> dict[str, float]:
    for row in rows:
        if row["scenario"] == scenario:
            return {policy: float(row[policy]) for policy in POLICIES}
    raise KeyError(scenario)


def unique_diagonal_count(rows: list[dict[str, str | float]]) -> int:
    count = 0
    for scenario in SCENARIOS:
        owner = DIAGONAL_POLICY[scenario]
        values = matrix_values(rows, scenario)
        if values[owner] - second_best(values, owner) > 0.0:
            count += 1
    return count


def outcome_saturation_count(success_rows: list[dict[str, str | float]]) -> int:
    count = 0
    for row in success_rows:
        if sum(float(row[policy]) >= 0.99 for policy in POLICIES) >= 2:
            count += 1
    return count


def candidate_score(evaluation: esas_eval.EsasEvaluation, pass_margin: float) -> dict[str, Any]:
    margins: dict[str, float] = {}
    diagonal_values: list[float] = []
    blocking: list[str] = []
    fields: dict[str, Any] = {}
    for scenario in SCENARIOS:
        owner = DIAGONAL_POLICY[scenario]
        values = matrix_values(evaluation.esas_rows, scenario)
        owner_value = values[owner]
        second = second_best(values, owner)
        margin = owner_value - second
        margins[scenario] = margin
        diagonal_values.append(owner_value)
        if margin <= 0.0:
            blocking.append(scenario)
        prefix = {
            "rear_close_threat": "rear_pi1",
            "flank_threat": "flank_pi2",
            "boundary_constrained": "boundary_pi3",
            "vertical_z_threat": "vertical_pi4",
        }[scenario]
        fields[f"{prefix}_esas"] = owner_value
        fields[f"{prefix.split('_')[0]}_second_best_esas"] = second
        fields[f"{prefix.split('_')[0]}_margin"] = margin
    unique_count = sum(value > 0.0 for value in margins.values())
    margin_pass_count = sum(value >= pass_margin for value in margins.values())
    min_margin = min(margins.values()) if margins else 0.0
    mean_diag = sum(diagonal_values) / len(diagonal_values) if diagonal_values else 0.0
    outcome_unique = unique_diagonal_count(evaluation.outcome_success_rows)
    saturation_count = outcome_saturation_count(evaluation.outcome_success_rows)
    tie_or_dominance_penalty = sum(max(0.0, pass_margin - value) for value in margins.values())
    model_selection_score = (
        10.0 * unique_count
        + 5.0 * margin_pass_count
        + 2.0 * min_margin
        + mean_diag
        - 2.0 * len(blocking)
        - 4.0 * tie_or_dominance_penalty
    )
    return {
        **fields,
        "esas_unique_diagonal_count": unique_count,
        "esas_margin_pass_count": margin_pass_count,
        "min_esas_margin": min_margin,
        "mean_diagonal_esas": mean_diag,
        "outcome_unique_diagonal_count": outcome_unique,
        "outcome_saturation_count": saturation_count,
        "model_selection_score": model_selection_score,
        "blocking_scenarios": ";".join(blocking),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_best_summary(path: Path, best: dict[str, Any] | None) -> None:
    if best is None:
        path.write_text("# ESAS Candidate Pair Selection\n\nNo candidate pair evaluated.\n", encoding="utf-8")
        return
    text = f"""# ESAS Candidate Pair Selection

Best pi1 candidate: `{best['pi1_candidate_path']}`

Best pi4 candidate: `{best['pi4_candidate_path']}`

Model selection score: {float(best['model_selection_score']):.4f}

ESAS unique diagonal count: {best['esas_unique_diagonal_count']}

ESAS margin pass count: {best['esas_margin_pass_count']}

Minimum ESAS margin: {float(best['min_esas_margin']):.4f}

Outcome saturation count: {best['outcome_saturation_count']}

Blocking scenarios: {best['blocking_scenarios'] or 'none'}

Terminal success is reported as a coarse outcome metric. Candidate selection is based on ESAS behavior-level specialization; terminal success saturation does not by itself reject a candidate.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best specialist candidate pair using ESAS")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--best-episodes", type=int, default=100)
    parser.add_argument("--candidate-dir", default="outputs/checkpoints/specialist_fix_candidates")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/paper_eval_core/esas_candidate_sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pass-margin", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    canonical_pi2 = checkpoint_dir / "sac_low_2_flank_threat.zip"
    canonical_pi3 = checkpoint_dir / "sac_low_3_boundary_constrained.zip"
    if not canonical_pi2.exists() or not canonical_pi3.exists():
        print(f"Missing checkpoint: {canonical_pi2 if not canonical_pi2.exists() else canonical_pi3}")
        print("Please run this selector locally with canonical pi2/pi3 checkpoints available.")
        return

    pi1_candidates, pi4_candidates = find_candidates(Path(args.candidate_dir))
    if not pi1_candidates or not pi4_candidates:
        print(f"No candidate pairs found under {args.candidate_dir}")
        return

    out_dir = Path(args.out_dir)
    score_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for pi1_path in pi1_candidates:
        for pi4_path in pi4_candidates:
            model_paths = [pi1_path, canonical_pi2, canonical_pi3, pi4_path]
            if args.skip_missing and any(not path.exists() for path in model_paths):
                continue
            pair_name = f"{pi1_path.stem}__{pi4_path.stem}"
            evaluation = esas_eval.evaluate_models(model_paths, episodes=args.episodes, seed=args.seed, device=args.device)
            esas_eval.write_matrix_csv(out_dir / "candidate_terminal_matrices" / f"{pair_name}.csv", evaluation.outcome_success_rows)
            esas_eval.write_matrix_csv(out_dir / "candidate_esas_matrices" / f"{pair_name}.csv", evaluation.esas_rows)
            score = candidate_score(evaluation, args.pass_margin)
            row = {
                "pi1_candidate_path": str(pi1_path),
                "pi4_candidate_path": str(pi4_path),
                "pi1_candidate_name": pi1_path.stem,
                "pi4_candidate_name": pi4_path.stem,
                **score,
            }
            score_rows.append(row)
            for component in evaluation.component_rows:
                component_rows.append({"candidate_pair": pair_name, **component})
            if best_row is None or float(row["model_selection_score"]) > float(best_row["model_selection_score"]):
                best_row = row

    score_rows.sort(key=lambda item: float(item["model_selection_score"]), reverse=True)
    write_rows(out_dir / "candidate_esas_scores.csv", score_rows)
    write_rows(out_dir / "candidate_component_breakdown.csv", component_rows)
    write_best_summary(out_dir / "best_candidate_summary.md", best_row)

    if best_row is not None:
        best_dir = out_dir / "best_100ep_eval"
        model_paths = [Path(best_row["pi1_candidate_path"]), canonical_pi2, canonical_pi3, Path(best_row["pi4_candidate_path"])]
        best_eval = esas_eval.evaluate_models(model_paths, episodes=args.best_episodes, seed=args.seed + 777_000, device=args.device)
        esas_eval.save_outputs(best_eval, best_dir, args.pass_margin)

    print("=== ESAS candidate selection complete ===")
    if best_row is None:
        print("No candidate pair evaluated.")
    else:
        print(f"best pi1 candidate path: {best_row['pi1_candidate_path']}")
        print(f"best pi4 candidate path: {best_row['pi4_candidate_path']}")
        print(f"model_selection_score: {float(best_row['model_selection_score']):.4f}")
        print(f"blocking_scenarios: {best_row['blocking_scenarios'] or 'none'}")
        print(f"best eval dir: {out_dir / 'best_100ep_eval'}")


if __name__ == "__main__":
    main()
