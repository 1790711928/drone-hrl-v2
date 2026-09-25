from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def detect_out_of_bounds_axis(
    x: float,
    y: float,
    z: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> str:
    if x < x_min:
        return "x_min"
    if x > x_max:
        return "x_max"
    if y < y_min:
        return "y_min"
    if y > y_max:
        return "y_max"
    if z < z_min:
        return "z_min"
    if z > z_max:
        return "z_max"
    return "none"


def maybe_plot_failed_trajectories(
    failed_episodes: list[dict[str, Any]],
    out_dir: Path,
    scenario: str,
    model_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed, skipping trajectory plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(failed_episodes[:3], start=1):
        xs = ep["xs"]
        ys = ep["ys"]
        zs = ep["zs"]
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, marker="o", markersize=2)
        ax.set_title(f"Fail Ep#{ep['episode']} ({ep['outcome']})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        out_path = out_dir / f"policy_behavior_traj_{scenario}_{model_name}_{i}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[plot] saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy behavior diagnostics for a single scenario/model")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", required=True, help="Path to SAC checkpoint")
    parser.add_argument("--episodes", type=int, default=30)
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
    model_name = model_path.stem.replace(".zip", "")

    rows: list[dict[str, Any]] = []
    failed_episodes: list[dict[str, Any]] = []
    axis_counts = {k: 0 for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "none"]}
    success_count = 0
    out_of_bounds_count = 0

    for ep in range(1, args.episodes + 1):
        env = PursuitEscapeGymEnv(scenario=args.scenario)
        obs, _ = env.reset()

        xs, ys, zs = [], [], []
        dists = []
        z_seps = []
        sum_abs_accel = 0.0
        sum_abs_yaw_rate = 0.0
        sum_abs_pitch_rate = 0.0
        max_abs_pitch = 0.0

        ep_reward = 0.0
        done = False
        info = {"outcome": "timeout", "distance": 0.0}

        while not done:
            state = env.inner.state
            if state is not None:
                xs.append(state.evader.x)
                ys.append(state.evader.y)
                zs.append(state.evader.z)
                dists.append(float(info.get("distance", 0.0)))
                z_seps.append(abs(state.evader.z - state.pursuer.z))
                max_abs_pitch = max(max_abs_pitch, abs(state.evader.pitch))

            action, _ = model.predict(obs, deterministic=True)
            action = action.tolist() if hasattr(action, "tolist") else list(action)
            sum_abs_accel += abs(float(action[0]))
            sum_abs_yaw_rate += abs(float(action[1]))
            sum_abs_pitch_rate += abs(float(action[2]))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        state = env.inner.state
        assert state is not None
        xs.append(state.evader.x)
        ys.append(state.evader.y)
        zs.append(state.evader.z)
        dists.append(float(info.get("distance", 0.0)))
        z_seps.append(abs(state.evader.z - state.pursuer.z))
        max_abs_pitch = max(max_abs_pitch, abs(state.evader.pitch))

        outcome = str(info.get("outcome", "timeout"))
        if outcome == "escaped":
            success_count += 1
        if outcome == "out_of_bounds":
            out_of_bounds_count += 1

        axis = detect_out_of_bounds_axis(
            x=state.evader.x,
            y=state.evader.y,
            z=state.evader.z,
            x_min=env.inner.term_cfg.x_min,
            x_max=env.inner.term_cfg.x_max,
            y_min=env.inner.term_cfg.y_min,
            y_max=env.inner.term_cfg.y_max,
            z_min=env.inner.term_cfg.z_min,
            z_max=env.inner.term_cfg.z_max,
        )
        if outcome != "out_of_bounds":
            axis = "none"
        axis_counts[axis] += 1

        steps = max(len(xs) - 1, 1)
        row = {
            "episode": ep,
            "outcome": outcome,
            "total_reward": ep_reward,
            "steps": steps,
            "final_x": xs[-1],
            "final_y": ys[-1],
            "final_z": zs[-1],
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
            "out_of_bounds_axis": axis,
            "min_boundary_margin": min(
                state.evader.x - env.inner.term_cfg.x_min,
                env.inner.term_cfg.x_max - state.evader.x,
                state.evader.y - env.inner.term_cfg.y_min,
                env.inner.term_cfg.y_max - state.evader.y,
                state.evader.z - env.inner.term_cfg.z_min,
                env.inner.term_cfg.z_max - state.evader.z,
            ),
            "min_distance": min(dists),
            "max_distance": max(dists),
            "avg_abs_accel": sum_abs_accel / steps,
            "avg_abs_yaw_rate": sum_abs_yaw_rate / steps,
            "avg_abs_pitch_rate": sum_abs_pitch_rate / steps,
            "max_abs_pitch": max_abs_pitch,
            "total_z_change": zs[-1] - zs[0],
            "max_z_separation": max(z_seps),
            "mean_z_separation": sum(z_seps) / len(z_seps),
        }
        rows.append(row)

        if outcome != "escaped":
            failed_episodes.append({"episode": ep, "outcome": outcome, "xs": xs, "ys": ys, "zs": zs})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"policy_behavior_{args.scenario}_{model_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = max(args.episodes, 1)
    avg_pitch_rate = sum(r["avg_abs_pitch_rate"] for r in rows) / n
    avg_yaw_rate = sum(r["avg_abs_yaw_rate"] for r in rows) / n
    avg_z_change = sum(r["total_z_change"] for r in rows) / n
    avg_min_margin = sum(r["min_boundary_margin"] for r in rows) / n

    print(f"[csv] saved: {out_csv}")
    print("=== Policy Behavior Summary ===")
    print(f"scenario={args.scenario}, model={model_name}, episodes={args.episodes}")
    print(f"success_rate={success_count / n:.3f}")
    print(f"out_of_bounds_rate={out_of_bounds_count / n:.3f}")
    print(f"out_of_bounds_axis_counts={axis_counts}")
    print(f"avg_abs_pitch_rate={avg_pitch_rate:.4f}")
    print(f"avg_abs_yaw_rate={avg_yaw_rate:.4f}")
    print(f"avg_z_change={avg_z_change:.4f}")
    print(f"avg_min_boundary_margin={avg_min_margin:.4f}")

    maybe_plot_failed_trajectories(failed_episodes, out_dir, args.scenario, model_name)


if __name__ == "__main__":
    main()
