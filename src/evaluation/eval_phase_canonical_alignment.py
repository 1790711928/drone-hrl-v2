from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation.eval_phase_option_discriminability import (
    MODEL_FILENAMES,
    OPTION_NAMES,
    PHASE_TYPES,
    _current_observation,
    evaluate_phase_option,
    reset_canonical_phase,
    reset_injected_phase,
)
from src.training.highlevel_env import HighLevelOptionEnv


CANONICAL_SCENARIO_BY_PHASE = {
    "rear": "rear_close_threat",
    "flank": "flank_threat",
    "boundary": "boundary_constrained",
    "vertical": "vertical_z_threat",
}
GEOMETRY_FIELDS = (
    "threat_forward",
    "threat_right",
    "threat_up",
    "distance",
    "min_boundary_margin",
    "evader_x_norm",
    "evader_y_norm",
    "evader_z_norm",
    "boundary_margin_x",
    "boundary_margin_y",
    "boundary_margin_z",
    "dx",
    "dy",
    "dz",
    "evader_pitch",
    "pursuer_pitch",
)
CSV_FIELDS = (
    "phase_type",
    "source",
    "canonical_scenario",
    "option",
    "improvement_score",
    "alignment_gap",
    *GEOMETRY_FIELDS,
)


def geometry_row(env: HighLevelOptionEnv, phase_type: str, source: str) -> dict[str, Any]:
    if source == "canonical":
        reset_canonical_phase(env, phase_type)
    else:
        reset_injected_phase(env, phase_type)
    obs = _current_observation(env)
    return {
        "phase_type": phase_type,
        "source": source,
        "canonical_scenario": CANONICAL_SCENARIO_BY_PHASE.get(phase_type, "composite-only: rear + vertical"),
        "option": "",
        "improvement_score": "",
        "alignment_gap": "",
        **{field: obs[field] for field in GEOMETRY_FIELDS},
    }


def print_geometry(rows: list[dict[str, Any]]) -> None:
    print("=== Canonical vs Injected Phase Geometry ===")
    for row in rows:
        print(
            f"{row['phase_type']:<14} {row['source']:<9} "
            f"tf={row['threat_forward']:+.3f} tr={row['threat_right']:+.3f} tu={row['threat_up']:+.3f} "
            f"dist={row['distance']:.3f} min_margin={row['min_boundary_margin']:.3f} "
            f"ex={row['evader_x_norm']:+.3f} ey={row['evader_y_norm']:+.3f} ez={row['evader_z_norm']:+.3f} "
            f"bx={row['boundary_margin_x']:.3f} by={row['boundary_margin_y']:.3f} bz={row['boundary_margin_z']:.3f} "
            f"dx={row['dx']:+.3f} dy={row['dy']:+.3f} dz={row['dz']:+.3f} "
            f"ep={row['evader_pitch']:+.3f} pp={row['pursuer_pitch']:+.3f}"
        )
    print("note: rear_vertical is composite-only and has no single canonical low-level scenario.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare canonical low-level scenarios with injected high-level phases")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--option-duration", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.option_duration <= 0:
        parser.error("--option-duration must be positive")

    # Geometry comparison is checkpoint-free. Placeholder models are sufficient
    # because reset and observation extraction do not execute an option.
    env = HighLevelOptionEnv(low_models=[None] * len(OPTION_NAMES), option_duration=args.option_duration, scenario_set="sequential")
    geometry_rows: list[dict[str, Any]] = []
    for phase_type in CANONICAL_SCENARIO_BY_PHASE:
        geometry_rows.append(geometry_row(env, phase_type, "canonical"))
        geometry_rows.append(geometry_row(env, phase_type, "injected"))
    geometry_rows.append(geometry_row(env, "rear_vertical", "injected"))
    print_geometry(geometry_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "phase_canonical_alignment.csv"

    checkpoint_dir = Path(args.checkpoint_dir)
    model_paths = [checkpoint_dir / filename for filename in MODEL_FILENAMES]
    missing_paths = [path for path in model_paths if not path.exists()]
    if missing_paths:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(geometry_rows)
        print(f"\nMissing checkpoint: {missing_paths[0]}")
        print("Geometry comparison completed. Run locally after training low-level SAC policies for option alignment matrices.")
        print(f"Saved geometry CSV: {output_path}")
        return

    from stable_baselines3 import SAC

    models = [SAC.load(str(path)) for path in model_paths]
    env = HighLevelOptionEnv(low_models=models, option_duration=args.option_duration, scenario_set="sequential")
    score_rows: list[dict[str, Any]] = []
    print("\n=== Canonical Phase × Option Improvement Matrix ===")
    for phase_type in CANONICAL_SCENARIO_BY_PHASE:
        canonical = [evaluate_phase_option(env, phase_type, option, args.episodes, "one_shot", source="canonical") for option in range(4)]
        injected = [evaluate_phase_option(env, phase_type, option, args.episodes, "one_shot", source="injected") for option in range(4)]
        print(f"{phase_type} canonical -> " + ", ".join(f"{row['option']}:{row['improvement_score']:+.3f}" for row in canonical))
        print(f"{phase_type} injected  -> " + ", ".join(f"{row['option']}:{row['improvement_score']:+.3f}" for row in injected))
        geometry_by_source = {source: geometry_row(env, phase_type, source) for source in ("canonical", "injected")}
        for canonical_row, injected_row in zip(canonical, injected):
            for source, result in (("canonical", canonical_row), ("injected", injected_row)):
                score_rows.append(
                    {
                        **geometry_by_source[source],
                        "option": result["option"],
                        "improvement_score": result["improvement_score"],
                        "alignment_gap": injected_row["improvement_score"] - canonical_row["improvement_score"],
                    }
                )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
