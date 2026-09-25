from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

BOUNDARY_DANGER_MARGIN = 0.15
BOUNDARY_CONTROLLABLE_MARGIN = 0.25
BOUNDARY_SAFE_MARGIN = 0.35
BOUNDARY_MARGIN_IMPROVE_STRONG = 0.12
BOUNDARY_MARGIN_IMPROVE_MIN = 0.08
BOUNDARY_DISTANCE_WORSE_LIMIT = -0.10


def detect_out_of_bounds_axis(state: Any, term_cfg: Any) -> str:
    if state.evader.x < term_cfg.x_min:
        return "x_min"
    if state.evader.x > term_cfg.x_max:
        return "x_max"
    if state.evader.y < term_cfg.y_min:
        return "y_min"
    if state.evader.y > term_cfg.y_max:
        return "y_max"
    if state.evader.z < term_cfg.z_min:
        return "z_min"
    if state.evader.z > term_cfg.z_max:
        return "z_max"
    return "none"


def maybe_plot_failed_trajectories(fails: list[dict[str, Any]], out_dir: Path, bounds: tuple[float, float, float, float, float, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed, skipping trajectory plots.")
        return

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(fails[:3], start=1):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ep["ev_xs"], ep["ev_ys"], ep["ev_zs"], marker="o", markersize=2, label="evader")
        ax.plot(ep["pu_xs"], ep["pu_ys"], ep["pu_zs"], linestyle="--", alpha=0.8, label="pursuer")
        ax.scatter([ep["ev_xs"][0]], [ep["ev_ys"][0]], [ep["ev_zs"][0]], marker="^", s=50, label="ev_start")
        ax.scatter([ep["ev_xs"][-1]], [ep["ev_ys"][-1]], [ep["ev_zs"][-1]], marker="s", s=50, label="ev_end")
        if ep["oob_axis"] != "none":
            ax.scatter([ep["ev_xs"][-1]], [ep["ev_ys"][-1]], [ep["ev_zs"][-1]], marker="x", s=80, color="red", label="oob")
        ax.set_title(f"Ep#{ep['episode']} outcome={ep['outcome']} axis={ep['oob_axis']}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.legend(loc="best", fontsize=8)
        out_path = out_dir / f"boundary_skill_fail_traj_{i}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[plot] saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="pi3 boundary skill failure diagnostics")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--skill-horizon", type=int, default=80)
    parser.add_argument("--model", default="outputs/checkpoints/sac_low_3_boundary_constrained.zip")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from src.training.sac_env import PursuitEscapeGymEnv

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Missing checkpoint: {model_path}")
        print("Please run this script locally after training low-level SAC policies.")
        return

    model = SAC.load(str(model_path))
    out_dir = Path(args.out_dir)

    rows: list[dict[str, Any]] = []
    failed_eps: list[dict[str, Any]] = []
    axis_counts = {k: 0 for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "none"]}
    bounds = None

    for ep in range(1, args.episodes + 1):
        env = PursuitEscapeGymEnv(scenario="boundary_constrained")
        obs, _ = env.reset()
        term_cfg = env.inner.term_cfg
        bounds = (term_cfg.x_min, term_cfg.x_max, term_cfg.y_min, term_cfg.y_max, term_cfg.z_min, term_cfg.z_max)

        start = env.inner._observation(closing_speed=0.0)
        start_bx = float(start["boundary_margin_x"])
        start_by = float(start["boundary_margin_y"])
        start_bz = float(start["boundary_margin_z"])
        start_min_margin = float(start["min_boundary_margin"])
        start_dist = float(start["distance"])

        done = False
        outcome = "running"
        completion_step = args.skill_horizon
        first_safe_region_step = -1
        skill_success = False
        leave_danger = start_min_margin >= BOUNDARY_DANGER_MARGIN
        reach_controllable = start_min_margin >= BOUNDARY_CONTROLLABLE_MARGIN
        reach_safe = start_min_margin >= BOUNDARY_SAFE_MARGIN

        sum_abs_accel = 0.0
        sum_abs_yaw = 0.0
        sum_abs_pitch = 0.0
        step_count = 0

        ev_xs: list[float] = []
        ev_ys: list[float] = []
        ev_zs: list[float] = []
        pu_xs: list[float] = []
        pu_ys: list[float] = []
        pu_zs: list[float] = []

        last = start
        while not done and step_count < args.skill_horizon:
            state = env.inner.state
            assert state is not None
            ev_xs.append(state.evader.x)
            ev_ys.append(state.evader.y)
            ev_zs.append(state.evader.z)
            pu_xs.append(state.pursuer.x)
            pu_ys.append(state.pursuer.y)
            pu_zs.append(state.pursuer.z)

            action, _ = model.predict(obs, deterministic=True)
            action = action.tolist() if hasattr(action, "tolist") else list(action)
            sum_abs_accel += abs(float(action[0]))
            sum_abs_yaw += abs(float(action[1]))
            sum_abs_pitch += abs(float(action[2]))

            obs, _, terminated, truncated, info = env.step(action)
            step_count += 1
            outcome = str(info.get("outcome", "running"))
            done = terminated or truncated

            last = env.inner._observation(closing_speed=float(info.get("closing_speed", 0.0)))
            cur_min_margin = float(last["min_boundary_margin"])
            dist_gain = float(last["distance"]) - start_dist

            leave_danger = leave_danger or (cur_min_margin >= BOUNDARY_DANGER_MARGIN)
            reach_controllable = reach_controllable or (cur_min_margin >= BOUNDARY_CONTROLLABLE_MARGIN)
            reach_safe = reach_safe or (cur_min_margin >= BOUNDARY_SAFE_MARGIN)

            if first_safe_region_step < 0 and cur_min_margin >= BOUNDARY_SAFE_MARGIN:
                first_safe_region_step = step_count

            strong_return = cur_min_margin >= BOUNDARY_SAFE_MARGIN
            recovered_not_danger = (cur_min_margin - start_min_margin) >= BOUNDARY_MARGIN_IMPROVE_STRONG and cur_min_margin >= BOUNDARY_DANGER_MARGIN
            controllable_recovery = cur_min_margin >= BOUNDARY_CONTROLLABLE_MARGIN and (cur_min_margin - start_min_margin) >= BOUNDARY_MARGIN_IMPROVE_MIN
            skill_success = (strong_return or controllable_recovery or recovered_not_danger) and dist_gain >= BOUNDARY_DISTANCE_WORSE_LIMIT and outcome != "out_of_bounds"
            if skill_success:
                completion_step = step_count
                break

        state = env.inner.state
        assert state is not None
        ev_xs.append(state.evader.x)
        ev_ys.append(state.evader.y)
        ev_zs.append(state.evader.z)
        pu_xs.append(state.pursuer.x)
        pu_ys.append(state.pursuer.y)
        pu_zs.append(state.pursuer.z)

        final_min_margin = float(last["min_boundary_margin"])
        min_margin_improve = final_min_margin - start_min_margin

        axis = detect_out_of_bounds_axis(state, term_cfg) if outcome == "out_of_bounds" else "none"
        axis_counts[axis] += 1

        steps = max(step_count, 1)
        row = {
            "episode": ep,
            "outcome": outcome,
            "skill_success": skill_success,
            "completion_step": completion_step,
            "final_x": state.evader.x,
            "final_y": state.evader.y,
            "final_z": state.evader.z,
            "min_x": min(ev_xs),
            "max_x": max(ev_xs),
            "min_y": min(ev_ys),
            "max_y": max(ev_ys),
            "min_z": min(ev_zs),
            "max_z": max(ev_zs),
            "out_of_bounds_axis": axis,
            "start_boundary_margin_x": start_bx,
            "start_boundary_margin_y": start_by,
            "start_boundary_margin_z": start_bz,
            "final_boundary_margin_x": float(last["boundary_margin_x"]),
            "final_boundary_margin_y": float(last["boundary_margin_y"]),
            "final_boundary_margin_z": float(last["boundary_margin_z"]),
            "min_boundary_margin_start": start_min_margin,
            "min_boundary_margin_final": final_min_margin,
            "min_boundary_margin_improvement": min_margin_improve,
            "leave_danger_zone": leave_danger,
            "reached_controllable_zone": reach_controllable,
            "reached_safe_region": reach_safe,
            "first_safe_region_step": first_safe_region_step,
            "distance_gain": float(last["distance"]) - start_dist,
            "avg_abs_accel": sum_abs_accel / steps,
            "avg_abs_yaw_rate": sum_abs_yaw / steps,
            "avg_abs_pitch_rate": sum_abs_pitch / steps,
            "handoff_after_boundary_recovery": skill_success,
        }
        rows.append(row)

        if not skill_success:
            failed_eps.append({"episode": ep, "outcome": outcome, "oob_axis": axis, "ev_xs": ev_xs, "ev_ys": ev_ys, "ev_zs": ev_zs, "pu_xs": pu_xs, "pu_ys": pu_ys, "pu_zs": pu_zs})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "boundary_skill_diagnostics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = max(len(rows), 1)
    success_rate = sum(1 for r in rows if bool(r["skill_success"])) / n
    leave_danger_rate = sum(1 for r in rows if bool(r["leave_danger_zone"])) / n
    controllable_rate = sum(1 for r in rows if bool(r["reached_controllable_zone"])) / n
    safe_rate = sum(1 for r in rows if bool(r["reached_safe_region"])) / n
    handoff_rate = sum(1 for r in rows if bool(r["handoff_after_boundary_recovery"])) / n
    oob_rate = sum(1 for r in rows if str(r["outcome"]) == "out_of_bounds") / n
    avg_margin_improve = sum(float(r["min_boundary_margin_improvement"]) for r in rows) / n
    avg_completion = sum(float(r["completion_step"]) for r in rows) / n
    avg_distance_gain = sum(float(r["distance_gain"]) for r in rows) / n
    x_max_oob_rate = axis_counts["x_max"] / n
    z_min_oob_rate = axis_counts["z_min"] / n

    print(f"[csv] saved: {out_csv}")
    print("=== Boundary Skill Summary (pi3 @ boundary_constrained) ===")
    print(f"boundary_recovery_success_rate={success_rate:.3f}")
    print(f"leave_danger_zone_rate={leave_danger_rate:.3f}")
    print(f"reach_controllable_zone_rate={controllable_rate:.3f}")
    print(f"reach_safe_zone_rate={safe_rate:.3f}")
    print(f"handoff_after_boundary_recovery_rate={handoff_rate:.3f}")
    print(f"out_of_bounds_rate={oob_rate:.3f}")
    print(f"out_of_bounds_axis_counts={axis_counts}")
    print(f"x_max_oob_rate={x_max_oob_rate:.3f}")
    print(f"z_min_oob_rate={z_min_oob_rate:.3f}")
    print(f"avg_min_boundary_margin_improvement={avg_margin_improve:.4f}")
    print(f"avg_completion_step={avg_completion:.2f}")
    print(f"avg_distance_gain={avg_distance_gain:.4f}")

    if bounds is not None:
        maybe_plot_failed_trajectories(failed_eps, out_dir, bounds)


if __name__ == "__main__":
    main()
