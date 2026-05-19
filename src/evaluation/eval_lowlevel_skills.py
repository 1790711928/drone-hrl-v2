from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation.eval_lowlevel_diagnostics import MODEL_FILENAMES, SCENARIOS


SKILL_ROWS = [
    ("pi1", "rear_close_threat", "sac_low_1_rear_close_threat.zip"),
    ("pi2", "flank_threat", "sac_low_2_flank_threat.zip"),
    ("pi3", "boundary_constrained", "sac_low_3_boundary_constrained.zip"),
    ("pi4", "vertical_z_threat", "sac_low_4_vertical_z_threat.zip"),
]


def run_episode(model: Any, scenario: str, skill_horizon: int) -> dict[str, float | str | bool]:
    from src.training.sac_env import PursuitEscapeGymEnv

    env = PursuitEscapeGymEnv(scenario=scenario)
    obs, _ = env.reset()
    start = dict(env.inner._observation(closing_speed=0.0))
    start_margin = min(start["boundary_margin_x"], start["boundary_margin_y"], start["boundary_margin_z"])
    start_dist = float(start["distance"])
    start_close = float(start["closing_speed"])
    start_threat_right = abs(float(start["threat_right"]))
    start_z_sep = abs(env.inner.state.evader.z - env.inner.state.pursuer.z)

    min_margin = start_margin
    min_dist = start_dist
    last = start
    outcome = "running"

    for _ in range(skill_horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        last = env.inner._observation(closing_speed=float(info.get("closing_speed", 0.0)))
        margin = min(last["boundary_margin_x"], last["boundary_margin_y"], last["boundary_margin_z"])
        min_margin = min(min_margin, margin)
        min_dist = min(min_dist, float(last["distance"]))
        outcome = str(info.get("outcome", "running"))
        if terminated or truncated:
            break

    state = env.inner.state
    assert state is not None
    end_dist = float(last["distance"])
    end_close = float(last["closing_speed"])
    end_threat_right = abs(float(last["threat_right"]))
    end_margin = min(last["boundary_margin_x"], last["boundary_margin_y"], last["boundary_margin_z"])
    end_z_sep = abs(state.evader.z - state.pursuer.z)

    return {
        "outcome": outcome,
        "distance_gain": end_dist - start_dist,
        "closing_speed_reduction": start_close - end_close,
        "capture_avoidance": outcome != "captured",
        "reached_safe_distance": end_dist > max(start_dist + 0.10, 0.55),
        "lateral_threat_reduction": start_threat_right - end_threat_right,
        "threat_right_abs_reduction": start_threat_right - end_threat_right,
        "min_boundary_margin_improvement": end_margin - start_margin,
        "return_to_safe_region": (start_margin < 0.30 and end_margin > 0.45) or (min_margin > 0.35),
        "out_of_bounds": outcome == "out_of_bounds",
        "vertical_separation_gain": end_z_sep - start_z_sep,
        "controlled_z_margin": float(last["boundary_margin_z"]) > 0.20,
        "z_out_of_bounds": outcome == "out_of_bounds" and (state.evader.z <= env.inner.term_cfg.z_min or state.evader.z >= env.inner.term_cfg.z_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level skill diagnostics on each policy's home scenario")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--skill-horizon", type=int, default=80)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    checkpoint_dir = Path(args.checkpoint_dir)
    # quick consistency check with official set
    for filename in MODEL_FILENAMES:
        path = checkpoint_dir / filename
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    rows: list[dict[str, Any]] = []
    for policy, scenario, filename in SKILL_ROWS:
        model = SAC.load(str(checkpoint_dir / filename))
        eps = [run_episode(model, scenario, args.skill_horizon) for _ in range(args.episodes)]
        n = max(len(eps), 1)

        def rate(key: str) -> float:
            return sum(1 for e in eps if bool(e[key])) / n

        def avg(key: str) -> float:
            return sum(float(e[key]) for e in eps) / n

        if policy == "pi1":
            skill_success_rate = sum(1 for e in eps if e["capture_avoidance"] and e["distance_gain"] > 0 and e["closing_speed_reduction"] > 0) / n
        elif policy == "pi2":
            skill_success_rate = sum(1 for e in eps if e["capture_avoidance"] and e["lateral_threat_reduction"] > 0 and e["distance_gain"] > 0) / n
        elif policy == "pi3":
            skill_success_rate = sum(1 for e in eps if (not e["out_of_bounds"]) and e["min_boundary_margin_improvement"] > 0) / n
        else:
            skill_success_rate = sum(1 for e in eps if e["vertical_separation_gain"] > 0 and e["controlled_z_margin"] and (not e["z_out_of_bounds"])) / n

        row = {
            "policy": policy,
            "scenario": scenario,
            "episodes": args.episodes,
            "skill_horizon": args.skill_horizon,
            "skill_success_rate": skill_success_rate,
            "distance_gain": avg("distance_gain"),
            "closing_speed_reduction": avg("closing_speed_reduction"),
            "capture_avoidance_rate": rate("capture_avoidance"),
            "reached_safe_distance_rate": rate("reached_safe_distance"),
            "lateral_threat_reduction": avg("lateral_threat_reduction"),
            "threat_right_abs_reduction": avg("threat_right_abs_reduction"),
            "min_boundary_margin_improvement": avg("min_boundary_margin_improvement"),
            "return_to_safe_region_rate": rate("return_to_safe_region"),
            "out_of_bounds_rate": rate("out_of_bounds"),
            "vertical_separation_gain": avg("vertical_separation_gain"),
            "controlled_z_margin_rate": rate("controlled_z_margin"),
            "z_out_of_bounds_rate": rate("z_out_of_bounds"),
        }
        rows.append(row)

        print(f"{policy}@{scenario}: skill_success_rate={skill_success_rate:.3f}, distance_gain={row['distance_gain']:.3f}, oob_rate={row['out_of_bounds_rate']:.3f}")

    out_csv = Path("outputs/evaluation/lowlevel_skill_diagnostics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] saved: {out_csv}")


if __name__ == "__main__":
    main()
