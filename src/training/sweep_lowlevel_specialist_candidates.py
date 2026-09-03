"""Candidate sweep for targeted low-level specialist correction.

The sweep starts every pi1/pi4 candidate from the original canonical checkpoint,
writes candidate checkpoints under specialist_fix_candidates, evaluates candidate
pairs with the soft 4x4 skill-alignment benchmark, and selects by unique diagonal
specialization rather than raw success alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import eval_lowlevel_skill_alignment as align

CANONICAL_CHECKPOINTS = [
    Path("outputs/checkpoints/sac_low_1_rear_close_threat.zip"),
    Path("outputs/checkpoints/sac_low_2_flank_threat.zip"),
    Path("outputs/checkpoints/sac_low_3_boundary_constrained.zip"),
    Path("outputs/checkpoints/sac_low_4_vertical_z_threat.zip"),
]
COMPONENT_KEYS = [
    "rear_distance_gain_component",
    "rear_closing_speed_component",
    "rear_safety_component",
    "rear_direction_consistency_component",
    "flank_threat_right_component",
    "flank_lateral_component",
    "flank_distance_component",
    "flank_safety_component",
    "boundary_margin_improvement_component",
    "boundary_final_margin_component",
    "boundary_safety_component",
    "vertical_separation_component",
    "vertical_threat_up_component",
    "vertical_z_safety_component",
    "vertical_distance_component",
]


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_seed_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("seed list must contain at least one integer")
    return values


def fine_tune_candidate(
    *,
    policy_id: int,
    scenario: str,
    input_checkpoint: Path,
    output_checkpoint: Path,
    timesteps: int,
    seed: int,
    device: str,
) -> None:
    if output_checkpoint.exists():
        print(f"[sweep] candidate exists, skipping training: {output_checkpoint}")
        return
    if not input_checkpoint.exists():
        print(f"Missing input checkpoint: {input_checkpoint}")
        print("Please run this sweep locally with trained canonical low-level SAC checkpoints available.")
        return

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for specialist candidate sweep.") from exc

    from src.training.sac_env import PursuitEscapeGymEnv

    env = DummyVecEnv(
        [
            lambda: PursuitEscapeGymEnv(
                scenario=scenario,
                scenario_weights=None,
                randomize_reset=True,
            )
        ]
    )
    env.seed(seed)
    print(
        f"[sweep] training pi{policy_id} candidate: scenario={scenario}, timesteps={timesteps}, "
        f"seed={seed}, output={output_checkpoint}"
    )
    model = SAC.load(str(input_checkpoint), env=env, device=device)
    model.set_random_seed(seed)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_checkpoint))


def margin(values: dict[str, float], owner: str) -> float:
    owner_value = values[owner]
    second_best = max(value for policy, value in values.items() if policy != owner)
    return owner_value - second_best


def unique_diagonal_count(rows: list[dict[str, str | float]]) -> tuple[int, float, dict[str, float]]:
    count = 0
    margins: dict[str, float] = {}
    for index, row in enumerate(rows):
        owner = align.POLICIES[index]
        values = {policy: float(row[policy]) for policy in align.POLICIES}
        current_margin = margin(values, owner)
        margins[str(row["scenario"])] = current_margin
        if current_margin > 0.0:
            count += 1
    min_margin = min(margins.values()) if margins else 0.0
    return count, min_margin, margins


def load_models(model_paths: list[Path]) -> list[Any]:
    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for sweep evaluation.") from exc
    return [SAC.load(str(path)) for path in model_paths]


def evaluate_pair(
    *,
    model_paths: list[Path],
    episodes: int,
    seed: int,
    pass_threshold: float,
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]], list[dict[str, str | float]], dict[tuple[str, str], align.AggregateSkillAlignment], list[dict[str, Any]]]:
    models = load_models(model_paths)
    aggregates: dict[tuple[str, str], align.AggregateSkillAlignment] = {}
    component_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(align.SCENARIOS):
        for policy_index, (policy, model) in enumerate(zip(align.POLICIES, models)):
            rollouts = []
            for episode_id in range(episodes):
                rollout_seed = seed + scenario_index * 100_000 + policy_index * 10_000 + episode_id
                rollouts.append(align.rollout_model(model, scenario, policy, episode_id, rollout_seed))
            aggregates[(scenario, policy)] = align.aggregate_rollouts(scenario, policy, rollouts, pass_threshold)
            component = {
                "scenario": scenario,
                "policy": policy,
                "episodes": episodes,
            }
            for key in COMPONENT_KEYS:
                component[key] = align.safe_mean([rollout.components.get(key, 0.0) for rollout in rollouts])
            component_rows.append(component)
    terminal_rows = align.matrix_rows(aggregates, "success_rate")
    skill_rows = align.matrix_rows(aggregates, "skill_alignment_score")
    pass_rows = align.matrix_rows(aggregates, "skill_alignment_pass_rate")
    return terminal_rows, skill_rows, pass_rows, aggregates, component_rows


def selection_score(terminal_rows: list[dict[str, str | float]], skill_rows: list[dict[str, str | float]], aggregates: dict[tuple[str, str], align.AggregateSkillAlignment]) -> dict[str, float]:
    terminal_unique_count, min_terminal_margin, terminal_margins = unique_diagonal_count(terminal_rows)
    skill_unique_count, min_skill_margin, skill_margins = unique_diagonal_count(skill_rows)

    rear_terminal = terminal_margins["rear_close_threat"]
    vertical_terminal = terminal_margins["vertical_z_threat"]
    rear_skill = skill_margins["rear_close_threat"]
    vertical_skill = skill_margins["vertical_z_threat"]
    flank_skill = skill_margins["flank_threat"]
    boundary_skill = skill_margins["boundary_constrained"]

    rear_pi1 = aggregates[("rear_close_threat", "pi1")]
    rear_pi2 = aggregates[("rear_close_threat", "pi2")]
    rear_pi4 = aggregates[("rear_close_threat", "pi4")]
    vertical_pi4 = aggregates[("vertical_z_threat", "pi4")]
    vertical_pi1 = aggregates[("vertical_z_threat", "pi1")]
    vertical_pi3 = aggregates[("vertical_z_threat", "pi3")]
    flank_pi2 = aggregates[("flank_threat", "pi2")]
    flank_pi1 = aggregates[("flank_threat", "pi1")]
    flank_pi4 = aggregates[("flank_threat", "pi4")]
    boundary_pi3 = aggregates[("boundary_constrained", "pi3")]

    penalty = 0.0
    penalty += max(0.0, 0.05 - rear_terminal) * 3.0
    penalty += max(0.0, 0.05 - vertical_terminal) * 3.0
    penalty += max(0.0, 0.05 - rear_skill) * 4.0
    penalty += max(0.0, 0.05 - vertical_skill) * 4.0
    penalty += max(0.0, 0.05 - flank_skill) * 3.0
    penalty += max(0.0, 0.05 - boundary_skill) * 3.0
    penalty += max(0.0, rear_pi2.success_rate - rear_pi1.success_rate + 0.05) * 2.0
    penalty += max(0.0, rear_pi4.success_rate - rear_pi1.success_rate + 0.02) * 1.0
    penalty += max(0.0, vertical_pi1.success_rate - vertical_pi4.success_rate + 0.05) * 2.0
    penalty += max(0.0, vertical_pi3.skill_alignment_score - vertical_pi4.skill_alignment_score + 0.05) * 3.0
    penalty += max(0.0, flank_pi1.skill_alignment_score - flank_pi2.skill_alignment_score + 0.05) * 3.0
    penalty += max(0.0, flank_pi4.skill_alignment_score - flank_pi2.skill_alignment_score + 0.05) * 2.0
    penalty += max(0.0, 0.05 - boundary_skill) * 4.0
    penalty += max(0.0, boundary_pi3.out_of_bounds_rate - 0.25) * 2.0
    penalty += (4 - terminal_unique_count) * 0.35
    penalty += (4 - skill_unique_count) * 0.60

    raw = (
        rear_pi1.success_rate
        + rear_pi1.skill_alignment_score
        + vertical_pi4.success_rate
        + vertical_pi4.skill_alignment_score
        + flank_skill
        + boundary_skill
        + 0.25 * terminal_unique_count
        + 0.35 * skill_unique_count
    )
    return {
        "model_selection_score": raw - penalty,
        "terminal_unique_diagonal_count": float(terminal_unique_count),
        "skill_unique_diagonal_count": float(skill_unique_count),
        "min_terminal_margin": min_terminal_margin,
        "min_skill_margin": min_skill_margin,
        "vertical_pi4_minus_second_terminal": vertical_terminal,
        "vertical_pi4_minus_second_skill": vertical_skill,
        "rear_pi1_minus_second_terminal": rear_terminal,
        "rear_pi1_minus_second_skill": rear_skill,
        "flank_pi2_minus_second_skill": flank_skill,
        "boundary_pi3_minus_second_skill": boundary_skill,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, rows: list[dict[str, str | float]]) -> None:
    align.write_matrix_csv(path, rows)


def write_best_summary(path: Path, best: dict[str, Any] | None) -> None:
    if best is None:
        path.write_text("# Specialist Candidate Sweep\n\nNo candidate was evaluated.\n", encoding="utf-8")
        return
    blocking = []
    for key in [
        "rear_pi1_minus_second_terminal",
        "rear_pi1_minus_second_skill",
        "vertical_pi4_minus_second_terminal",
        "vertical_pi4_minus_second_skill",
        "flank_pi2_minus_second_skill",
        "boundary_pi3_minus_second_skill",
    ]:
        if float(best[key]) <= 0.0:
            blocking.append(key)
    passes = not blocking and int(best["terminal_unique_diagonal_count"]) == 4 and int(best["skill_unique_diagonal_count"]) == 4
    lines = [
        "# Specialist Candidate Sweep",
        "",
        f"Best pi1 candidate: `{best['pi1_candidate_path']}`",
        f"Best pi4 candidate: `{best['pi4_candidate_path']}`",
        f"Model selection score: {float(best['model_selection_score']):.4f}",
        f"Terminal unique diagonal count: {best['terminal_unique_diagonal_count']}",
        f"Skill unique diagonal count: {best['skill_unique_diagonal_count']}",
        f"Strict diagonal highest passes: {passes}",
        f"Blocking margins: {', '.join(blocking) if blocking else 'none'}",
        "",
        "Tied highest performance is treated as failure because each diagonal option must be strictly higher than all non-diagonal options.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pi1 rear and pi4 vertical specialist fine-tune candidates")
    parser.add_argument("--pi1-timesteps", type=parse_int_list, default=parse_int_list("1000,2000,5000,10000,20000,30000"))
    parser.add_argument("--pi4-timesteps", type=parse_int_list, default=parse_int_list("1000,2000,5000,10000,20000,30000,50000"))
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--best-episodes", type=int, default=100)
    parser.add_argument("--pass-threshold", type=float, default=0.60)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--candidate-dir", default="outputs/checkpoints/specialist_fix_candidates")
    parser.add_argument("--out-dir", default="outputs/paper_eval_core/specialist_candidate_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0 or args.best_episodes <= 0:
        parser.error("--episodes and --best-episodes must be positive")

    checkpoint_dir = Path(args.checkpoint_dir)
    canonical = [checkpoint_dir / path.name for path in CANONICAL_CHECKPOINTS]
    missing = [path for path in canonical if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing checkpoint: {path}")
        print("Please run this sweep locally after training canonical low-level SAC policies.")
        return

    candidate_dir = Path(args.candidate_dir)
    pi1_candidates: list[tuple[Path, int, int]] = []
    pi4_candidates: list[tuple[Path, int, int]] = []

    for seed in args.seeds:
        for timesteps in args.pi1_timesteps:
            out = candidate_dir / "pi1_rear" / f"t{timesteps}_seed{seed}.zip"
            if not args.skip_training:
                fine_tune_candidate(
                    policy_id=1,
                    scenario="rear_close_threat",
                    input_checkpoint=canonical[0],
                    output_checkpoint=out,
                    timesteps=timesteps,
                    seed=seed,
                    device=args.device,
                )
            pi1_candidates.append((out, timesteps, seed))
        for timesteps in args.pi4_timesteps:
            out = candidate_dir / "pi4_vertical" / f"t{timesteps}_seed{seed}.zip"
            if not args.skip_training:
                fine_tune_candidate(
                    policy_id=4,
                    scenario="vertical_z_threat",
                    input_checkpoint=canonical[3],
                    output_checkpoint=out,
                    timesteps=timesteps,
                    seed=seed,
                    device=args.device,
                )
            pi4_candidates.append((out, timesteps, seed))

    out_dir = Path(args.out_dir)
    terminal_dir = out_dir / "candidate_terminal_matrices"
    skill_dir = out_dir / "candidate_skill_score_matrices"
    pass_dir = out_dir / "candidate_pass_matrices"
    scores: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for pi1_path, pi1_timesteps, pi1_seed in pi1_candidates:
        for pi4_path, pi4_timesteps, pi4_seed in pi4_candidates:
            if not pi1_path.exists() or not pi4_path.exists():
                print(f"[sweep] skipping unevaluated pair, missing candidate: {pi1_path} or {pi4_path}")
                continue
            pair_name = f"pi1_t{pi1_timesteps}_s{pi1_seed}__pi4_t{pi4_timesteps}_s{pi4_seed}"
            model_paths = [pi1_path, canonical[1], canonical[2], pi4_path]
            terminal_rows, skill_rows, pass_rows, aggregates, components = evaluate_pair(
                model_paths=model_paths,
                episodes=args.episodes,
                seed=pi1_seed * 1_000_000 + pi4_seed * 10_000,
                pass_threshold=args.pass_threshold,
            )
            write_matrix(terminal_dir / f"{pair_name}.csv", terminal_rows)
            write_matrix(skill_dir / f"{pair_name}.csv", skill_rows)
            write_matrix(pass_dir / f"{pair_name}.csv", pass_rows)

            score_bits = selection_score(terminal_rows, skill_rows, aggregates)
            row: dict[str, Any] = {
                "candidate_pair": pair_name,
                "pi1_candidate_path": str(pi1_path),
                "pi4_candidate_path": str(pi4_path),
                "pi1_timesteps": pi1_timesteps,
                "pi4_timesteps": pi4_timesteps,
                "pi1_seed": pi1_seed,
                "pi4_seed": pi4_seed,
                **score_bits,
            }
            scores.append(row)
            for component in components:
                component_rows.append({"candidate_pair": pair_name, **component})
            if best_row is None or float(row["model_selection_score"]) > float(best_row["model_selection_score"]):
                best_row = row

    scores.sort(key=lambda item: float(item["model_selection_score"]), reverse=True)
    write_rows(out_dir / "candidate_scores.csv", scores)
    write_rows(out_dir / "component_breakdown.csv", component_rows)
    write_best_summary(out_dir / "best_candidate_summary.md", best_row)

    if best_row is not None:
        best_dir = out_dir / "best_100ep_eval"
        model_paths = [Path(best_row["pi1_candidate_path"]), canonical[1], canonical[2], Path(best_row["pi4_candidate_path"])]
        terminal_rows, skill_rows, pass_rows, aggregates, components = evaluate_pair(
            model_paths=model_paths,
            episodes=args.best_episodes,
            seed=777_000,
            pass_threshold=args.pass_threshold,
        )
        write_matrix(best_dir / "terminal_success_matrix.csv", terminal_rows)
        write_matrix(best_dir / "skill_alignment_score_matrix.csv", skill_rows)
        write_matrix(best_dir / "skill_alignment_pass_matrix.csv", pass_rows)
        align.write_summary(best_dir / "skill_alignment_summary.md", terminal_rows, skill_rows, pass_rows, args.best_episodes, args.pass_threshold)

    print("=== Specialist candidate sweep complete ===")
    if best_row is None:
        print("No candidate pair was evaluated.")
    else:
        print(f"best pi1 candidate path: {best_row['pi1_candidate_path']}")
        print(f"best pi4 candidate path: {best_row['pi4_candidate_path']}")
        print(f"best model_selection_score: {float(best_row['model_selection_score']):.4f}")
        print(f"terminal_unique_diagonal_count: {best_row['terminal_unique_diagonal_count']}")
        print(f"skill_unique_diagonal_count: {best_row['skill_unique_diagonal_count']}")
        print(f"best 100ep outputs: {out_dir / 'best_100ep_eval'}")


if __name__ == "__main__":
    main()
