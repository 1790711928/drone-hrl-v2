"""Score reconstructed raw rows with the explicit Chapter 3 formula.

No figure or heatmap is generated here. All normalization bounds and validity
choices come from the independent YAML configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml


POLICIES = ["pi1", "pi2", "pi3", "pi4"]
RAW_FIELDS = ["M_turn", "G_v", "G_s", "R_s", "G_b", "R_b", "G_z", "G_d"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def refuse_nonempty_output(path: Path) -> None:
    resolved = path.resolve()
    if BASE_DIR.resolve() not in resolved.parents:
        raise ValueError(f"output must remain under {BASE_DIR}, got {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def normalize(value: float, lower: float, upper: float, clip: bool) -> float:
    if upper <= lower:
        raise ValueError(f"normalization upper must exceed lower: {lower}, {upper}")
    result = (value - lower) / (upper - lower)
    return max(0.0, min(1.0, result)) if clip else result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "metric_config.yaml")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    refuse_nonempty_output(args.output_dir)

    scenario_specs = {str(spec["id"]): spec for spec in config["scenarios"]}
    bounds = config["normalization"]["bounds"]
    clip = bool(config["normalization"].get("clip", True))
    validity = config["validity"]
    if validity.get("mode") != "escaped_only":
        raise ValueError("this reconstruction currently supports only validity.mode=escaped_only")
    escaped_value = float(validity["escaped_value"])
    failed_value = float(validity["failed_value"])

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV contains no episode rows")
    missing = set(RAW_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"input CSV is missing raw fields: {sorted(missing)}")

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        scenario = row["scenario"]
        if scenario not in scenario_specs:
            raise ValueError(f"scenario is not defined in config: {scenario}")
        metric_pair = [str(value) for value in scenario_specs[scenario]["metric_pair"]]
        normalized: dict[str, float] = {}
        for metric in RAW_FIELDS:
            spec = bounds[metric]
            normalized[f"{metric}_bar"] = normalize(
                float(row[metric]), float(spec["lower"]), float(spec["upper"]), clip
            )
        v = escaped_value if row["outcome"] == "escaped" else failed_value
        scored = dict(row)
        scored.update(normalized)
        scored["m1_name"] = metric_pair[0]
        scored["m2_name"] = metric_pair[1]
        scored["V"] = v
        scored["score"] = v * ((normalized[f"{metric_pair[0]}_bar"] + normalized[f"{metric_pair[1]}_bar"]) / 2.0)
        scored_rows.append(scored)

    scored_fields = list(scored_rows[0].keys())
    scored_path = args.output_dir / "episodes_scored.csv"
    with scored_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scored_fields)
        writer.writeheader()
        writer.writerows(scored_rows)

    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        cell_rows[(row["scenario"], row["policy"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for scenario in scenario_specs:
        for policy in POLICIES:
            cell = cell_rows.get((scenario, policy), [])
            if not cell:
                raise ValueError(f"missing policy/scenario cell: {policy}/{scenario}")
            summary: dict[str, Any] = {
                "scenario": scenario,
                "scenario_label": scenario_specs[scenario]["label"],
                "policy": policy,
                "target_policy": scenario_specs[scenario]["target_policy"],
                "metric_1": scenario_specs[scenario]["metric_pair"][0],
                "metric_2": scenario_specs[scenario]["metric_pair"][1],
                "episode_count": len(cell),
                "mean_V": mean([float(row["V"]) for row in cell]),
                "mean_score": mean([float(row["score"]) for row in cell]),
            }
            for metric in RAW_FIELDS:
                summary[f"mean_{metric}"] = mean([float(row[metric]) for row in cell])
                summary[f"mean_{metric}_bar"] = mean([float(row[f"{metric}_bar"]) for row in cell])
            summary_rows.append(summary)

    summary_path = args.output_dir / "metric_summary.csv"
    summary_fields = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    matrix_path = args.output_dir / "score_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", *POLICIES])
        writer.writeheader()
        for scenario in scenario_specs:
            values = {row["policy"]: row["mean_score"] for row in summary_rows if row["scenario"] == scenario}
            writer.writerow({"scenario": scenario, **{policy: values[policy] for policy in POLICIES}})

    manifest = {
        "schema": "reconstructed_metric_system_v1",
        "status": "score_matrix_complete_no_heatmap",
        "git_commit_at_run": git_commit(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "input_csv": str(args.input_csv.resolve()),
        "input_csv_sha256": sha256_file(args.input_csv),
        "validity_mode": validity["mode"],
        "normalization_is_cross_policy_fixed": True,
        "outputs": [str(path.resolve()) for path in [scored_path, summary_path, matrix_path]],
        "heatmap_generated": False,
    }
    with (args.output_dir / "score_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved scored episodes to {scored_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved 4x4 score matrix to {matrix_path}")
    print("No heatmap was generated.")


if __name__ == "__main__":
    main()
