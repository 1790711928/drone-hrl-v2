from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import eval_lowlevel_skill_alignment as rollout_lib
from src.evaluation.escape_skill_alignment import ESAS_COMPONENTS, compute_all_esas_scores, compute_esas_for_scenario

SCENARIOS = rollout_lib.SCENARIOS
POLICIES = rollout_lib.POLICIES
MODEL_FILENAMES = rollout_lib.MODEL_FILENAMES
DIAGONAL_POLICY = dict(zip(SCENARIOS, POLICIES))
PANEL_TITLES = {
    ("rear_close_threat", "pi1"): "Rear escape option",
    ("flank_threat", "pi2"): "Flank evasion option",
    ("boundary_constrained", "pi3"): "Boundary recovery option",
    ("vertical_z_threat", "pi4"): "Vertical escape option",
}
COMPONENT_FIELDS = [field for skill in ("rear", "flank", "boundary", "vertical") for field in ESAS_COMPONENTS[skill]]


@dataclass
class EsasEvaluation:
    outcome_success_rows: list[dict[str, str | float]]
    outcome_reward_rows: list[dict[str, str | float]]
    outcome_capture_rows: list[dict[str, str | float]]
    outcome_oob_rows: list[dict[str, str | float]]
    esas_rows: list[dict[str, str | float]]
    component_rows: list[dict[str, Any]]
    best_rollouts: dict[tuple[str, str], rollout_lib.RolloutSkillAlignment]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.policy_checkpoints is not None:
        return [Path(value) for value in args.policy_checkpoints]
    checkpoint_dir = Path(args.checkpoint_dir)
    return [
        Path(args.policy1_checkpoint) if args.policy1_checkpoint else checkpoint_dir / MODEL_FILENAMES[0],
        Path(args.policy2_checkpoint) if args.policy2_checkpoint else checkpoint_dir / MODEL_FILENAMES[1],
        Path(args.policy3_checkpoint) if args.policy3_checkpoint else checkpoint_dir / MODEL_FILENAMES[2],
        Path(args.policy4_checkpoint) if args.policy4_checkpoint else checkpoint_dir / MODEL_FILENAMES[3],
    ]


def load_models(model_paths: list[Path], device: str) -> list[Any]:
    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required to load SAC checkpoints.") from exc
    return [SAC.load(str(path), device=device) for path in model_paths]


def matrix_from_values(values: dict[tuple[str, str], float]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for scenario in SCENARIOS:
        row: dict[str, str | float] = {"scenario": scenario}
        for policy in POLICIES:
            row[policy] = values[(scenario, policy)]
        rows.append(row)
    return rows


def evaluate_models(model_paths: list[Path], *, episodes: int, seed: int, device: str) -> EsasEvaluation:
    models = load_models(model_paths, device)
    success: dict[tuple[str, str], float] = {}
    reward: dict[tuple[str, str], float] = {}
    capture: dict[tuple[str, str], float] = {}
    oob: dict[tuple[str, str], float] = {}
    selected_esas: dict[tuple[str, str], float] = {}
    component_rows: list[dict[str, Any]] = []
    best_rollouts: dict[tuple[str, str], rollout_lib.RolloutSkillAlignment] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        for policy_index, (policy, model) in enumerate(zip(POLICIES, models)):
            rollouts = []
            for episode_id in range(episodes):
                rollout_seed = seed + scenario_index * 100_000 + policy_index * 10_000 + episode_id
                rollouts.append(rollout_lib.rollout_model(model, scenario, policy, episode_id, rollout_seed))
            key = (scenario, policy)
            success[key] = sum(r.outcome == "escaped" for r in rollouts) / max(len(rollouts), 1)
            reward[key] = safe_mean([r.total_reward for r in rollouts])
            capture[key] = sum(r.outcome == "captured" for r in rollouts) / max(len(rollouts), 1)
            oob[key] = sum(r.outcome == "out_of_bounds" for r in rollouts) / max(len(rollouts), 1)
            components = {field: safe_mean([r.components.get(field, 0.0) for r in rollouts]) for field in COMPONENT_FIELDS}
            esas_scores = compute_all_esas_scores(components)
            selected = compute_esas_for_scenario(scenario, components)
            selected_esas[key] = selected
            component_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "episodes": episodes,
                    **components,
                    **esas_scores,
                    "selected_esas_for_scenario": selected,
                }
            )
            best_rollouts[key] = max(rollouts, key=lambda rollout: compute_esas_for_scenario(scenario, rollout.components))

    return EsasEvaluation(
        outcome_success_rows=matrix_from_values(success),
        outcome_reward_rows=matrix_from_values(reward),
        outcome_capture_rows=matrix_from_values(capture),
        outcome_oob_rows=matrix_from_values(oob),
        esas_rows=matrix_from_values(selected_esas),
        component_rows=component_rows,
        best_rollouts=best_rollouts,
    )


def write_matrix_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", *POLICIES])
        writer.writeheader()
        writer.writerows(rows)


def write_component_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario",
        "policy",
        "episodes",
        *COMPONENT_FIELDS,
        "rear_esas",
        "flank_esas",
        "boundary_esas",
        "vertical_esas",
        "selected_esas_for_scenario",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(rows: list[dict[str, str | float]], png_path: Path, pdf_path: Path, title: str, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[heatmap] matplotlib not installed; skipped {png_path}")
        return
    data = np.array([[float(row[policy]) for policy in POLICIES] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(POLICIES)), labels=POLICIES)
    ax.set_yticks(range(len(SCENARIOS)), labels=SCENARIOS)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Scenario")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def save_representative_panel(best_rollouts: dict[tuple[str, str], rollout_lib.RolloutSkillAlignment], png_path: Path, pdf_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[panel] matplotlib not installed; skipped representative ESAS panel.")
        return
    pairs = [
        ("rear_close_threat", "pi1"),
        ("flank_threat", "pi2"),
        ("boundary_constrained", "pi3"),
        ("vertical_z_threat", "pi4"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, pair in zip(axes.flat, pairs):
        rollout = best_rollouts.get(pair)
        ax.set_title(PANEL_TITLES[pair])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        if rollout is None:
            ax.text(0.5, 0.5, "No rollout", transform=ax.transAxes, ha="center", va="center")
            continue
        ev_x = [point[0] for point in rollout.evader_points]
        ev_y = [point[1] for point in rollout.evader_points]
        pu_x = [point[0] for point in rollout.pursuer_points]
        pu_y = [point[1] for point in rollout.pursuer_points]
        ax.plot(ev_x, ev_y, color="tab:blue", linewidth=2.2, label="evader")
        ax.plot(pu_x, pu_y, color="tab:red", linestyle="--", alpha=0.55, linewidth=1.4, label="pursuer")
        ax.scatter(ev_x[0], ev_y[0], color="green", s=35, label="start", zorder=4)
        ax.scatter(ev_x[-1], ev_y[-1], color="black", marker="x", s=45, label="end", zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def row_values(row: dict[str, str | float]) -> dict[str, float]:
    return {policy: float(row[policy]) for policy in POLICIES}


def scenario_margin(row: dict[str, str | float]) -> tuple[str, float, str, float, float]:
    scenario = str(row["scenario"])
    owner = DIAGONAL_POLICY[scenario]
    values = row_values(row)
    owner_value = values[owner]
    others = [(policy, value) for policy, value in values.items() if policy != owner]
    best_policy, best_value = max(values.items(), key=lambda item: item[1])
    second_best = max(value for _, value in others)
    return owner, owner_value, best_policy, best_value, owner_value - second_best


def outcome_saturation(success_rows: list[dict[str, str | float]]) -> tuple[bool, int]:
    count = 0
    for row in success_rows:
        saturated = [policy for policy in POLICIES if float(row[policy]) >= 0.99]
        if len(saturated) >= 2:
            count += 1
    return count > 0, count


def write_summary(path: Path, evaluation: EsasEvaluation, pass_margin: float) -> dict[str, Any]:
    saturation, saturation_count = outcome_saturation(evaluation.outcome_success_rows)
    strict_pass = True
    margin_pass = True
    blocking: list[str] = []
    lines = [
        "# Escape Skill Alignment Score (ESAS) Low-level Specialization",
        "",
        "ESAS_k(policy, scenario) = sum_i w_k,i * component_k,i, clipped to [0, 1].",
        "ESAS is a behavior-level metric: it measures whether the intended escape skill is expressed.",
        "Outcome metrics are coarse outcomes and are not the primary specialization evidence.",
        "Terminal success measures whether the evader survives, while ESAS measures whether the intended escape skill is expressed.",
        "",
        f"Outcome saturation: {saturation} (scenario_count={saturation_count})",
        "",
        "## ESAS diagonal dominance",
        "",
    ]
    margins: dict[str, float] = {}
    for row in evaluation.esas_rows:
        scenario = str(row["scenario"])
        owner, owner_value, best_policy, best_value, margin = scenario_margin(row)
        second_best = owner_value - margin
        margins[scenario] = margin
        if best_policy != owner or margin <= 0.0:
            strict_pass = False
            blocking.append(scenario)
        if margin < pass_margin:
            margin_pass = False
        evidence = "strong" if margin >= pass_margin else ("weak positive" if margin > 0 else "failed")
        lines.append(
            f"- `{scenario}`: diagonal `{owner}`={owner_value:.3f}; best `{best_policy}`={best_value:.3f}; "
            f"second_best={second_best:.3f}; margin={margin:.3f}; evidence={evidence}."
        )
    lines.extend(
        [
            "",
            f"Strict diagonal highest: {strict_pass}",
            f"All margins >= {pass_margin:.3f}: {margin_pass}",
            f"Low-level ESAS specialization passes: {strict_pass and margin_pass}",
            f"Blocking scenarios: {', '.join(blocking) if blocking else 'none'}",
            "",
            "## Paper-use conclusion",
            "",
            "Use ESAS as the main low-level specialization evidence. Report terminal success, reward, capture, and out-of-bounds as outcome-level context only.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "outcome_saturation": saturation,
        "outcome_saturation_count": saturation_count,
        "strict_diagonal_highest": strict_pass,
        "margin_pass": margin_pass,
        "blocking_scenarios": blocking,
        "margins": margins,
    }


def save_outputs(evaluation: EsasEvaluation, out_dir: Path, pass_margin: float) -> dict[str, Any]:
    write_matrix_csv(out_dir / "outcome_success_matrix.csv", evaluation.outcome_success_rows)
    write_matrix_csv(out_dir / "outcome_reward_matrix.csv", evaluation.outcome_reward_rows)
    write_matrix_csv(out_dir / "outcome_capture_matrix.csv", evaluation.outcome_capture_rows)
    write_matrix_csv(out_dir / "outcome_out_of_bounds_matrix.csv", evaluation.outcome_oob_rows)
    write_matrix_csv(out_dir / "esas_matrix.csv", evaluation.esas_rows)
    write_component_csv(out_dir / "esas_component_breakdown.csv", evaluation.component_rows)
    save_heatmap(evaluation.outcome_success_rows, out_dir / "outcome_success_heatmap.png", out_dir / "outcome_success_heatmap.pdf", "Outcome Success Matrix", "success rate")
    save_heatmap(evaluation.esas_rows, out_dir / "esas_heatmap.png", out_dir / "esas_heatmap.pdf", "ESAS Matrix", "ESAS")
    save_representative_panel(evaluation.best_rollouts, out_dir / "representative_esas_skills_panel.png", out_dir / "representative_esas_skills_panel.pdf")
    return write_summary(out_dir / "esas_summary.md", evaluation, pass_margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate low-level Escape Skill Alignment Score (ESAS)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/paper_eval_core/esas_lowlevel_specialization")
    parser.add_argument("--policy-checkpoints", nargs=4, default=None, metavar=("PI1", "PI2", "PI3", "PI4"))
    parser.add_argument("--policy1-checkpoint", default="")
    parser.add_argument("--policy2-checkpoint", default="")
    parser.add_argument("--policy3-checkpoint", default="")
    parser.add_argument("--policy4-checkpoint", default="")
    parser.add_argument("--pass-margin", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model_paths = resolve_model_paths(args)
    for path in model_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this ESAS evaluation locally after training low-level SAC policies.")
            return
    evaluation = evaluate_models(model_paths, episodes=args.episodes, seed=args.seed, device=args.device)
    out_dir = Path(args.out_dir)
    summary = save_outputs(evaluation, out_dir, args.pass_margin)
    print("=== ESAS matrix ===")
    for row in evaluation.esas_rows:
        print(row)
    print("=== Outcome success matrix ===")
    for row in evaluation.outcome_success_rows:
        print(row)
    print(f"Strict diagonal highest: {summary['strict_diagonal_highest']}")
    print(f"Outcome saturation: {summary['outcome_saturation']}")
    print(f"Blocking scenarios: {summary['blocking_scenarios']}")


if __name__ == "__main__":
    main()
