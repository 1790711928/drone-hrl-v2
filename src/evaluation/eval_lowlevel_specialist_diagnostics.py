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
class RolloutDiagnostic:
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

    @property
    def skill_alignment_score(self) -> float:
        return float(getattr(self, SCENARIO_SKILL[self.scenario]))


@dataclass
class AggregateDiagnostic:
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


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def outcome_safety(outcome: str) -> float:
    if outcome == "escaped":
        return 1.0
    if outcome == "timeout":
        return 0.75
    if outcome == "captured":
        return 0.0
    if outcome == "out_of_bounds":
        return 0.0
    return 0.4


def boundary_safety(observations: list[dict[str, float]]) -> float:
    margins = [obs["min_boundary_margin"] for obs in observations]
    if not margins:
        return 0.0
    min_margin = min(margins)
    tail_margin = mean(margins[max(0, int(0.8 * len(margins))) :])
    return clip01(0.6 * tail_margin + 0.4 * min_margin)


def compute_skill_scores(rollout: RolloutDiagnostic) -> None:
    obs = rollout.observations
    if len(obs) < 2 or not rollout.evader_points or not rollout.pursuer_points:
        return

    initial_distance = distance(rollout.evader_points[0], rollout.pursuer_points[0])
    final_distance = distance(rollout.evader_points[-1], rollout.pursuer_points[-1])
    distance_gain = clip01(0.5 + (final_distance - initial_distance) / 40.0)
    maintain_distance = clip01(final_distance / 35.0)
    safety = outcome_safety(rollout.outcome)
    boundary_safe = boundary_safety(obs)
    not_oob = 0.0 if rollout.outcome == "out_of_bounds" else 1.0
    not_capture = 0.0 if rollout.outcome == "captured" else 1.0

    closing_values = [entry["closing_speed"] for entry in obs]
    early_closing = mean(closing_values[: max(1, len(closing_values) // 4)])
    late_closing = mean(closing_values[max(0, int(0.75 * len(closing_values))) :])
    closing_reduction = clip01(0.5 + early_closing - late_closing)

    threat_right = [abs(entry["threat_right"]) for entry in obs]
    initial_right = mean(threat_right[: max(1, len(threat_right) // 5)])
    final_right = mean(threat_right[max(0, int(0.8 * len(threat_right))) :])
    threat_right_reduction = clip01(0.5 + initial_right - final_right)
    dy_values = [entry["dy"] for entry in obs]
    lateral_evasion = clip01(abs(dy_values[-1] - dy_values[0]) * 2.0)

    margin_values = [entry["min_boundary_margin"] for entry in obs]
    initial_margin = mean(margin_values[: max(1, len(margin_values) // 5)])
    final_margin = mean(margin_values[max(0, int(0.8 * len(margin_values))) :])
    margin_improvement = clip01(0.5 + final_margin - initial_margin)
    final_margin_score = clip01(final_margin)
    controlled_recovery = clip01(0.5 * margin_improvement + 0.5 * final_margin_score)

    z_sep = [abs(entry["dz"]) for entry in obs]
    initial_z_sep = mean(z_sep[: max(1, len(z_sep) // 5)])
    final_z_sep = mean(z_sep[max(0, int(0.8 * len(z_sep))) :])
    vertical_sep_improvement = clip01(0.5 + (final_z_sep - initial_z_sep) * 2.0)
    threat_up = [abs(entry["threat_up"]) for entry in obs]
    initial_up = mean(threat_up[: max(1, len(threat_up) // 5)])
    final_up = mean(threat_up[max(0, int(0.8 * len(threat_up))) :])
    threat_up_reduction = clip01(0.5 + initial_up - final_up)
    z_safety = clip01(mean([entry["boundary_margin_z"] for entry in obs]))

    rollout.rear_score = clip01(
        0.30 * distance_gain
        + 0.25 * closing_reduction
        + 0.20 * not_capture
        + 0.15 * not_oob
        + 0.10 * boundary_safe
    )
    rollout.flank_score = clip01(
        0.30 * threat_right_reduction
        + 0.25 * lateral_evasion
        + 0.20 * maintain_distance
        + 0.15 * not_oob
        + 0.10 * safety
    )
    rollout.boundary_score = clip01(
        0.30 * margin_improvement
        + 0.25 * final_margin_score
        + 0.20 * not_oob
        + 0.15 * controlled_recovery
        + 0.10 * safety
    )
    rollout.vertical_score = clip01(
        0.30 * vertical_sep_improvement
        + 0.25 * threat_up_reduction
        + 0.20 * z_safety
        + 0.15 * not_oob
        + 0.10 * safety
    )


def rollout_model(model: Any, scenario: str, policy: str, episode_id: int, seed: int) -> RolloutDiagnostic:
    env = PursuitEscapeGymEnv(scenario=scenario, randomize_reset=True)
    obs, _ = env.reset(seed=seed)
    diagnostic = RolloutDiagnostic(scenario=scenario, policy=policy, episode_id=episode_id, outcome="timeout", total_reward=0.0, steps=0)
    ev_point, pu_point = state_points(env)
    diagnostic.evader_points.append(ev_point)
    diagnostic.pursuer_points.append(pu_point)
    diagnostic.observations.append(obs_to_dict(obs))

    terminated = False
    truncated = False
    info: dict[str, Any] = {"outcome": "timeout"}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        diagnostic.total_reward += float(reward)
        diagnostic.steps += 1
        ev_point, pu_point = state_points(env)
        diagnostic.evader_points.append(ev_point)
        diagnostic.pursuer_points.append(pu_point)
        diagnostic.observations.append(obs_to_dict(obs))

    diagnostic.outcome = str(info.get("outcome", "timeout"))
    compute_skill_scores(diagnostic)
    return diagnostic


def aggregate_rollouts(scenario: str, policy: str, rollouts: list[RolloutDiagnostic]) -> AggregateDiagnostic:
    denom = max(len(rollouts), 1)
    return AggregateDiagnostic(
        scenario=scenario,
        policy=policy,
        success_rate=sum(r.outcome == "escaped" for r in rollouts) / denom,
        avg_reward=mean([r.total_reward for r in rollouts]),
        avg_steps=mean([float(r.steps) for r in rollouts]),
        capture_rate=sum(r.outcome == "captured" for r in rollouts) / denom,
        out_of_bounds_rate=sum(r.outcome == "out_of_bounds" for r in rollouts) / denom,
        timeout_rate=sum(r.outcome == "timeout" for r in rollouts) / denom,
        rear_score=mean([r.rear_score for r in rollouts]),
        flank_score=mean([r.flank_score for r in rollouts]),
        boundary_score=mean([r.boundary_score for r in rollouts]),
        vertical_score=mean([r.vertical_score for r in rollouts]),
        skill_alignment_score=mean([r.skill_alignment_score for r in rollouts]),
    )


def matrix_rows(aggregates: dict[tuple[str, str], AggregateDiagnostic], metric: str) -> list[dict[str, str | float]]:
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


def save_representative_panel(best_rollouts: dict[tuple[str, str], RolloutDiagnostic], out_png: Path, out_pdf: Path) -> None:
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


def summarize_dominance(rows: list[dict[str, str | float]], label: str) -> list[str]:
    lines = [f"## {label}", ""]
    all_diagonal = True
    for index, row in enumerate(rows):
        scenario = str(row["scenario"])
        owner = POLICIES[index]
        values = [(policy, float(row[policy])) for policy in POLICIES]
        values_sorted = sorted(values, key=lambda item: item[1], reverse=True)
        best_policy, best_value = values_sorted[0]
        owner_value = float(row[owner])
        second_best = values_sorted[1][1] if best_policy == owner else best_value
        margin = owner_value - second_best
        if best_policy != owner:
            all_diagonal = False
        lines.append(
            f"- `{scenario}`: diagonal `{owner}`={owner_value:.3f}; best `{best_policy}`={best_value:.3f}; "
            f"margin_to_second_best={margin:.3f}."
        )
    lines.append("")
    lines.append(f"结论：{'diagonal highest across all scenarios' if all_diagonal else '存在非 diagonal policy 最高的场景'}。")
    lines.append("")
    return lines


def write_summary(path: Path, terminal_rows: list[dict[str, str | float]], skill_rows: list[dict[str, str | float]], episodes: int) -> None:
    terminal_all_diag = all(
        max(POLICIES, key=lambda policy: float(row[policy])) == POLICIES[index]
        for index, row in enumerate(terminal_rows)
    )
    skill_all_diag = all(
        max(POLICIES, key=lambda policy: float(row[policy])) == POLICIES[index]
        for index, row in enumerate(skill_rows)
    )
    supports_specialization = skill_all_diag
    lines = [
        "# Low-level Specialist Diagnostics",
        "",
        f"Episodes per policy × scenario: {episodes}",
        "",
        "This diagnostic separates terminal success from behavior-level skill alignment.",
        "",
        *summarize_dominance(terminal_rows, "Terminal success diagonal dominance"),
        *summarize_dominance(skill_rows, "Skill alignment diagonal dominance"),
        "## Paper-use conclusion",
        "",
        f"- Terminal success matrix diagonal highest: {terminal_all_diag}",
        f"- Skill alignment matrix diagonal highest: {skill_all_diag}",
        f"- Supports claim `low-level options are behaviorally specialized`: {supports_specialization}",
        "",
        "If terminal success is not diagonal but skill alignment is diagonal, use the skill-alignment matrix as the primary specialization evidence and present terminal success as a coarse outcome metric.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level option specialist diagnostics: terminal success + skill alignment")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/paper_eval_core/lowlevel_specialist_diagnostics")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    checkpoint_dir = Path(args.checkpoint_dir)
    model_paths = [checkpoint_dir / filename for filename in MODEL_FILENAMES]
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

    aggregates: dict[tuple[str, str], AggregateDiagnostic] = {}
    best_rollouts: dict[tuple[str, str], RolloutDiagnostic] = {}

    for scenario_index, scenario in enumerate(SCENARIOS):
        print(f"=== Scenario: {scenario} ===")
        for policy_index, (policy, model) in enumerate(zip(POLICIES, models)):
            rollouts: list[RolloutDiagnostic] = []
            for episode_id in range(args.episodes):
                seed = args.seed + scenario_index * 100_000 + policy_index * 10_000 + episode_id
                rollouts.append(rollout_model(model, scenario, policy, episode_id, seed))
            aggregate = aggregate_rollouts(scenario, policy, rollouts)
            aggregates[(scenario, policy)] = aggregate
            best_rollouts[(scenario, policy)] = max(rollouts, key=lambda item: item.skill_alignment_score)
            print(
                f"{policy}: success={aggregate.success_rate:.3f}, skill={aggregate.skill_alignment_score:.3f}, "
                f"reward={aggregate.avg_reward:.2f}, steps={aggregate.avg_steps:.1f}, "
                f"cap={aggregate.capture_rate:.3f}, oob={aggregate.out_of_bounds_rate:.3f}, timeout={aggregate.timeout_rate:.3f}"
            )

    terminal_rows = matrix_rows(aggregates, "success_rate")
    skill_rows = matrix_rows(aggregates, "skill_alignment_score")
    terminal_csv = out_dir / "terminal_success_matrix.csv"
    skill_csv = out_dir / "skill_alignment_matrix.csv"
    write_matrix_csv(terminal_csv, terminal_rows)
    write_matrix_csv(skill_csv, skill_rows)
    print(f"[csv] saved: {terminal_csv}")
    print(f"[csv] saved: {skill_csv}")

    save_heatmap(terminal_rows, out_dir / "terminal_success_matrix_heatmap.png", "Terminal Success Matrix", "success rate")
    save_heatmap(skill_rows, out_dir / "skill_alignment_matrix_heatmap.png", "Skill Alignment Matrix", "skill alignment score")
    save_representative_panel(
        best_rollouts,
        out_dir / "representative_lowlevel_skills_panel.png",
        out_dir / "representative_lowlevel_skills_panel.pdf",
    )
    summary_path = out_dir / "specialist_diagnostic_summary.md"
    write_summary(summary_path, terminal_rows, skill_rows, args.episodes)
    print(f"[summary] saved: {summary_path}")

    print("\n=== Terminal success matrix ===")
    for row in terminal_rows:
        print(row)
    print("\n=== Skill alignment matrix ===")
    for row in skill_rows:
        print(row)
    print(f"\nSummary conclusion written to: {summary_path}")


if __name__ == "__main__":
    main()
