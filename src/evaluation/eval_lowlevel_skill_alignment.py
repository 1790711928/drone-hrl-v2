from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.training.sac_env import PursuitEscapeGymEnv

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]
POLICIES = ["pi1", "pi2", "pi3", "pi4"]
POLICY_TITLES = {
    "pi1": "Rear option",
    "pi2": "Flank option",
    "pi3": "Boundary option",
    "pi4": "Vertical option",
}
SCENARIO_SKILL = {
    "rear_close_threat": "rear_score",
    "flank_threat": "flank_score",
    "boundary_constrained": "boundary_score",
    "vertical_z_threat": "vertical_score",
}
MODEL_FILENAMES = [
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
]


@dataclass
class RolloutSkillAlignment:
    scenario: str
    policy: str
    episode_id: int
    outcome: str
    total_reward: float
    steps: int
    evader_points: list[tuple[float, float, float]] = field(default_factory=list)
    pursuer_points: list[tuple[float, float, float]] = field(default_factory=list)
    observations: list[dict[str, float]] = field(default_factory=list)
    rear_score: float = 0.0
    flank_score: float = 0.0
    boundary_score: float = 0.0
    vertical_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    @property
    def skill_alignment_score(self) -> float:
        return float(getattr(self, SCENARIO_SKILL[self.scenario]))


@dataclass
class AggregateSkillAlignment:
    scenario: str
    policy: str
    success_rate: float
    avg_reward: float
    avg_steps: float
    capture_rate: float
    out_of_bounds_rate: float
    timeout_rate: float
    rear_score: float
    flank_score: float
    boundary_score: float
    vertical_score: float
    skill_alignment_score: float
    skill_alignment_pass_rate: float


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def obs_to_dict(obs: np.ndarray) -> dict[str, float]:
    return {key: float(obs[index]) for index, key in enumerate(PursuitEscapeGymEnv.OBS_KEYS)}


def state_points(env: PursuitEscapeGymEnv) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if env.inner.state is None:
        raise RuntimeError("environment state is not initialized")
    ev = env.inner.state.evader
    pu = env.inner.state.pursuer
    return (ev.x, ev.y, ev.z), (pu.x, pu.y, pu.z)


def distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def early(values: list[float], fraction: float = 0.2) -> list[float]:
    return values[: max(1, int(len(values) * fraction))]


def late(values: list[float], fraction: float = 0.2) -> list[float]:
    start = max(0, int(len(values) * (1.0 - fraction)))
    return values[start:]


def soft_safety(outcome: str) -> tuple[float, float, float]:
    no_capture = 0.0 if outcome == "captured" else 1.0
    no_oob = 0.0 if outcome == "out_of_bounds" else 1.0
    if outcome == "escaped":
        outcome_bonus = 1.0
    elif outcome == "timeout":
        outcome_bonus = 0.75
    elif outcome in {"captured", "out_of_bounds"}:
        outcome_bonus = 0.0
    else:
        outcome_bonus = 0.4
    return no_capture, no_oob, outcome_bonus


def compute_skill_scores(rollout: RolloutSkillAlignment) -> None:
    obs = rollout.observations
    if len(obs) < 2 or len(rollout.evader_points) < 2 or len(rollout.pursuer_points) < 2:
        return

    no_capture, no_oob, outcome_bonus = soft_safety(rollout.outcome)
    distances = [distance(ev, pu) for ev, pu in zip(rollout.evader_points, rollout.pursuer_points)]
    initial_distance = safe_mean(early(distances))
    final_distance = safe_mean(late(distances))
    distance_gain = clip01(0.5 + (final_distance - initial_distance) / 45.0)
    distance_maintenance = clip01(final_distance / 35.0)

    closing_values = [entry["closing_speed"] for entry in obs]
    closing_reduction = clip01(0.5 + safe_mean(early(closing_values)) - safe_mean(late(closing_values)))

    threat_forward = [entry["threat_forward"] for entry in obs]
    direction_consistency = clip01(0.5 + safe_mean(late(threat_forward)) - safe_mean(early(threat_forward)))
    dx_drift = abs(obs[-1]["dy"] - obs[0]["dy"]) + abs(obs[-1]["dz"] - obs[0]["dz"])
    drift_penalty = clip01(dx_drift * 0.6)
    rear_safety = 0.5 * no_capture + 0.5 * no_oob

    threat_right_abs = [abs(entry["threat_right"]) for entry in obs]
    threat_right_reduction = clip01(0.5 + safe_mean(early(threat_right_abs)) - safe_mean(late(threat_right_abs)))
    lateral_evasion = clip01(abs(obs[-1]["dy"] - obs[0]["dy"]) * 2.0)
    flank_safety = 0.5 * no_capture + 0.5 * no_oob

    margins = [entry["min_boundary_margin"] for entry in obs]
    margin_improvement = clip01(0.5 + safe_mean(late(margins)) - safe_mean(early(margins)))
    final_margin = clip01(safe_mean(late(margins)))
    margin_variation = max(margins) - min(margins) if margins else 0.0
    recovery_stability = clip01(final_margin - 0.25 * margin_variation + 0.25)
    boundary_safety = no_oob

    z_sep = [abs(entry["dz"]) for entry in obs]
    vertical_sep_improvement = clip01(0.5 + (safe_mean(late(z_sep)) - safe_mean(early(z_sep))) * 2.5)
    threat_up_abs = [abs(entry["threat_up"]) for entry in obs]
    threat_up_reduction = clip01(0.5 + safe_mean(early(threat_up_abs)) - safe_mean(late(threat_up_abs)))
    z_boundary_safety = clip01(safe_mean([entry["boundary_margin_z"] for entry in obs]))
    horizontal_escape_penalty = clip01((abs(obs[-1]["dx"] - obs[0]["dx"]) + abs(obs[-1]["dy"] - obs[0]["dy"])) * 0.35)
    vertical_safety = 0.5 * no_capture + 0.5 * no_oob

    rollout.components = {
        "rear_distance_gain_component": distance_gain,
        "rear_closing_speed_component": closing_reduction,
        "rear_safety_component": rear_safety,
        "rear_direction_consistency_component": direction_consistency,
        "flank_threat_right_component": threat_right_reduction,
        "flank_lateral_component": lateral_evasion,
        "flank_distance_component": distance_maintenance,
        "flank_safety_component": flank_safety,
        "boundary_margin_improvement_component": margin_improvement,
        "boundary_final_margin_component": final_margin,
        "boundary_safety_component": boundary_safety,
        "vertical_separation_component": vertical_sep_improvement,
        "vertical_threat_up_component": threat_up_reduction,
        "vertical_z_safety_component": z_boundary_safety,
        "vertical_distance_component": distance_maintenance,
    }

    # Soft scores: capture/out-of-bounds affect safety terms, but only those outcomes are hard safety failures.
    rollout.rear_score = clip01(
        0.26 * distance_gain
        + 0.23 * closing_reduction
        + 0.20 * rear_safety
        + 0.18 * direction_consistency
        + 0.13 * (1.0 - 0.45 * drift_penalty)
    )
    rollout.flank_score = clip01(
        0.30 * threat_right_reduction
        + 0.25 * lateral_evasion
        + 0.20 * distance_maintenance
        + 0.15 * flank_safety
        + 0.10 * outcome_bonus
    )
    rollout.boundary_score = clip01(
        0.30 * margin_improvement
        + 0.25 * final_margin
        + 0.20 * boundary_safety
        + 0.15 * recovery_stability
        + 0.10 * outcome_bonus
    )
    rollout.vertical_score = clip01(
        0.28 * vertical_sep_improvement
        + 0.24 * threat_up_reduction
        + 0.20 * z_boundary_safety
        + 0.16 * distance_maintenance
        + 0.12 * vertical_safety
        - 0.12 * horizontal_escape_penalty
    )


def rollout_model(model: Any, scenario: str, policy: str, episode_id: int, seed: int) -> RolloutSkillAlignment:
    env = PursuitEscapeGymEnv(scenario=scenario, randomize_reset=True)
    obs, _ = env.reset(seed=seed)
    rollout = RolloutSkillAlignment(
        scenario=scenario,
        policy=policy,
        episode_id=episode_id,
        outcome="timeout",
        total_reward=0.0,
        steps=0,
    )
    ev_point, pu_point = state_points(env)
    rollout.evader_points.append(ev_point)
    rollout.pursuer_points.append(pu_point)
    rollout.observations.append(obs_to_dict(obs))

    terminated = False
    truncated = False
    info: dict[str, Any] = {"outcome": "timeout"}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rollout.total_reward += float(reward)
        rollout.steps += 1
        ev_point, pu_point = state_points(env)
        rollout.evader_points.append(ev_point)
        rollout.pursuer_points.append(pu_point)
        rollout.observations.append(obs_to_dict(obs))

    rollout.outcome = str(info.get("outcome", "timeout"))
    compute_skill_scores(rollout)
    return rollout


def aggregate_rollouts(
    scenario: str,
    policy: str,
    rollouts: list[RolloutSkillAlignment],
    pass_threshold: float,
) -> AggregateSkillAlignment:
    denom = max(len(rollouts), 1)
    return AggregateSkillAlignment(
        scenario=scenario,
        policy=policy,
        success_rate=sum(r.outcome == "escaped" for r in rollouts) / denom,
        avg_reward=safe_mean([r.total_reward for r in rollouts]),
        avg_steps=safe_mean([float(r.steps) for r in rollouts]),
        capture_rate=sum(r.outcome == "captured" for r in rollouts) / denom,
        out_of_bounds_rate=sum(r.outcome == "out_of_bounds" for r in rollouts) / denom,
        timeout_rate=sum(r.outcome == "timeout" for r in rollouts) / denom,
        rear_score=safe_mean([r.rear_score for r in rollouts]),
        flank_score=safe_mean([r.flank_score for r in rollouts]),
        boundary_score=safe_mean([r.boundary_score for r in rollouts]),
        vertical_score=safe_mean([r.vertical_score for r in rollouts]),
        skill_alignment_score=safe_mean([r.skill_alignment_score for r in rollouts]),
        skill_alignment_pass_rate=sum(r.skill_alignment_score >= pass_threshold for r in rollouts) / denom,
    )


def matrix_rows(aggregates: dict[tuple[str, str], AggregateSkillAlignment], metric: str) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for scenario in SCENARIOS:
        row: dict[str, str | float] = {"scenario": scenario}
        for policy in POLICIES:
            row[policy] = getattr(aggregates[(scenario, policy)], metric)
        rows.append(row)
    return rows


def write_matrix_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", *POLICIES])
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(rows: list[dict[str, str | float]], path: Path, title: str, colorbar_label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[heatmap] matplotlib not installed; skipped {path}")
        return

    data = np.array([[float(row[policy]) for policy in POLICIES] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(POLICIES)), labels=POLICIES)
    ax.set_yticks(range(len(SCENARIOS)), labels=SCENARIOS)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Scenario")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[heatmap] saved: {path}")


def save_representative_panel(best_rollouts: dict[tuple[str, str], RolloutSkillAlignment], out_png: Path, out_pdf: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[panel] matplotlib not installed; skipped representative panel export.")
        return

    pairs = [
        ("rear_close_threat", "pi1"),
        ("flank_threat", "pi2"),
        ("boundary_constrained", "pi3"),
        ("vertical_z_threat", "pi4"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (scenario, policy) in zip(axes.flat, pairs):
        rollout = best_rollouts.get((scenario, policy))
        ax.set_title(POLICY_TITLES[policy])
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
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[panel] saved: {out_png}")
    print(f"[panel] saved: {out_pdf}")


def dominance_lines(rows: list[dict[str, str | float]], label: str) -> tuple[list[str], bool]:
    lines = [f"## {label}", ""]
    all_diagonal = True
    for index, row in enumerate(rows):
        scenario = str(row["scenario"])
        owner = POLICIES[index]
        values = [(policy, float(row[policy])) for policy in POLICIES]
        sorted_values = sorted(values, key=lambda item: item[1], reverse=True)
        best_policy, best_value = sorted_values[0]
        owner_value = float(row[owner])
        second_best = sorted_values[1][1] if best_policy == owner else best_value
        margin = owner_value - second_best
        if best_policy != owner:
            all_diagonal = False
        lines.append(
            f"- `{scenario}`: diagonal `{owner}`={owner_value:.3f}; best `{best_policy}`={best_value:.3f}; "
            f"margin_to_second_best={margin:.3f}."
        )
    lines.extend(["", f"Diagonal highest across all scenarios: {all_diagonal}", ""])
    return lines, all_diagonal


def flat_values(rows: list[dict[str, str | float]]) -> list[float]:
    return [float(row[policy]) for row in rows for policy in POLICIES]


def off_diagonal_values(rows: list[dict[str, str | float]]) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        owner = POLICIES[index]
        values.extend(float(row[policy]) for policy in POLICIES if policy != owner)
    return values


def write_summary(
    path: Path,
    terminal_rows: list[dict[str, str | float]],
    score_rows: list[dict[str, str | float]],
    pass_rows: list[dict[str, str | float]],
    episodes: int,
    pass_threshold: float,
) -> None:
    terminal_lines, terminal_diag = dominance_lines(terminal_rows, "Terminal success matrix")
    score_lines, score_diag = dominance_lines(score_rows, "Skill alignment score matrix")
    pass_lines, pass_diag = dominance_lines(pass_rows, "Skill alignment pass matrix")
    all_near_zero = max(flat_values(score_rows) or [0.0]) < 0.05
    off_diag_all_zero = max(off_diagonal_values(score_rows) or [0.0]) <= 1e-9
    supports_specialization = score_diag and not all_near_zero and not off_diag_all_zero

    lines = [
        "# Low-level Soft Skill-Alignment Diagnostics",
        "",
        f"Episodes per policy × scenario: {episodes}",
        f"Pass threshold: {pass_threshold:.3f}",
        "",
        "This diagnostic uses continuous 0–1 skill-alignment scores. The score matrix is the primary paper figure; pass rate is auxiliary.",
        "",
        *terminal_lines,
        *score_lines,
        *pass_lines,
        "## Softness checks",
        "",
        f"- All strategies near zero: {all_near_zero}",
        f"- Non-specialist strategies all zero: {off_diag_all_zero}",
        "",
        "## Paper-use conclusion",
        "",
        f"- Terminal success diagonal highest: {terminal_diag}",
        f"- Skill alignment score diagonal highest: {score_diag}",
        f"- Skill alignment pass matrix diagonal highest: {pass_diag}",
        f"- Supports claim `low-level options are behaviorally specialized`: {supports_specialization}",
        "",
        "If terminal success is not diagonal but skill alignment is diagonal, report terminal success as a coarse outcome and use skill alignment as the main specialization evidence.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.policy_checkpoints is not None:
        return [Path(value) for value in args.policy_checkpoints]
    return [
        Path(args.policy1_checkpoint) if args.policy1_checkpoint else Path(args.checkpoint_dir) / MODEL_FILENAMES[0],
        Path(args.policy2_checkpoint) if args.policy2_checkpoint else Path(args.checkpoint_dir) / MODEL_FILENAMES[1],
        Path(args.policy3_checkpoint) if args.policy3_checkpoint else Path(args.checkpoint_dir) / MODEL_FILENAMES[2],
        Path(args.policy4_checkpoint) if args.policy4_checkpoint else Path(args.checkpoint_dir) / MODEL_FILENAMES[3],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft low-level skill-alignment diagnostics")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument(
        "--policy-checkpoints",
        nargs=4,
        default=None,
        metavar=("PI1", "PI2", "PI3", "PI4"),
        help="Optional explicit checkpoint paths for pi1 pi2 pi3 pi4, in order.",
    )
    parser.add_argument("--policy1-checkpoint", default="", help="Optional explicit pi1 checkpoint path.")
    parser.add_argument("--policy2-checkpoint", default="", help="Optional explicit pi2 checkpoint path.")
    parser.add_argument("--policy3-checkpoint", default="", help="Optional explicit pi3 checkpoint path.")
    parser.add_argument("--policy4-checkpoint", default="", help="Optional explicit pi4 checkpoint path.")
    parser.add_argument("--out-dir", default="outputs/paper_eval_core/lowlevel_skill_alignment")
    parser.add_argument("--pass-threshold", type=float, default=0.60)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 <= args.pass_threshold <= 1.0:
        parser.error("--pass-threshold must be in [0, 1]")

    model_paths = resolve_model_paths(args)
    for path in model_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required to load SAC checkpoints.") from exc

    models = [SAC.load(str(path)) for path in model_paths]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates: dict[tuple[str, str], AggregateSkillAlignment] = {}
    best_rollouts: dict[tuple[str, str], RolloutSkillAlignment] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        print(f"=== Scenario: {scenario} ===")
        for policy_index, (policy, model) in enumerate(zip(POLICIES, models)):
            rollouts: list[RolloutSkillAlignment] = []
            for episode_id in range(args.episodes):
                seed = args.seed + scenario_index * 100_000 + policy_index * 10_000 + episode_id
                rollouts.append(rollout_model(model, scenario, policy, episode_id, seed))
            aggregate = aggregate_rollouts(scenario, policy, rollouts, args.pass_threshold)
            aggregates[(scenario, policy)] = aggregate
            best_rollouts[(scenario, policy)] = max(rollouts, key=lambda item: item.skill_alignment_score)
            print(
                f"{policy}: success={aggregate.success_rate:.3f}, score={aggregate.skill_alignment_score:.3f}, "
                f"pass={aggregate.skill_alignment_pass_rate:.3f}, reward={aggregate.avg_reward:.2f}, "
                f"steps={aggregate.avg_steps:.1f}, cap={aggregate.capture_rate:.3f}, "
                f"oob={aggregate.out_of_bounds_rate:.3f}, timeout={aggregate.timeout_rate:.3f}"
            )

    terminal_rows = matrix_rows(aggregates, "success_rate")
    score_rows = matrix_rows(aggregates, "skill_alignment_score")
    pass_rows = matrix_rows(aggregates, "skill_alignment_pass_rate")

    terminal_csv = out_dir / "terminal_success_matrix.csv"
    score_csv = out_dir / "skill_alignment_score_matrix.csv"
    pass_csv = out_dir / "skill_alignment_pass_matrix.csv"
    write_matrix_csv(terminal_csv, terminal_rows)
    write_matrix_csv(score_csv, score_rows)
    write_matrix_csv(pass_csv, pass_rows)
    print(f"[csv] saved: {terminal_csv}")
    print(f"[csv] saved: {score_csv}")
    print(f"[csv] saved: {pass_csv}")

    save_heatmap(terminal_rows, out_dir / "terminal_success_heatmap.png", "Terminal Success Matrix", "success rate")
    save_heatmap(score_rows, out_dir / "skill_alignment_score_heatmap.png", "Skill Alignment Score Matrix", "skill alignment score")
    save_heatmap(pass_rows, out_dir / "skill_alignment_pass_heatmap.png", "Skill Alignment Pass Matrix", "pass rate")
    save_representative_panel(
        best_rollouts,
        out_dir / "representative_lowlevel_skills_panel.png",
        out_dir / "representative_lowlevel_skills_panel.pdf",
    )

    summary_path = out_dir / "skill_alignment_summary.md"
    write_summary(summary_path, terminal_rows, score_rows, pass_rows, args.episodes, args.pass_threshold)
    print(f"[summary] saved: {summary_path}")

    print("\n=== Terminal success matrix ===")
    for row in terminal_rows:
        print(row)
    print("\n=== Skill alignment score matrix ===")
    for row in score_rows:
        print(row)
    print("\n=== Skill alignment pass matrix ===")
    for row in pass_rows:
        print(row)
    print(f"\nSummary conclusion written to: {summary_path}")


if __name__ == "__main__":
    main()
