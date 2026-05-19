from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation.eval_lowlevel_diagnostics import MODEL_FILENAMES

SKILL_ROWS = [
    ("pi1", "rear_close_threat", "sac_low_1_rear_close_threat.zip"),
    ("pi2", "flank_threat", "sac_low_2_flank_threat.zip"),
    ("pi3", "boundary_constrained", "sac_low_3_boundary_constrained.zip"),
    ("pi4", "vertical_z_threat", "sac_low_4_vertical_z_threat.zip"),
]

REAR_DIST_GAIN_MIN = 0.08
REAR_SAFE_DISTANCE = 0.56
REAR_CLOSE_RELIEF_MAX = 0.05

FLANK_THREAT_REDUCTION_MIN = 0.15
FLANK_SAFE_THREAT_RIGHT_ABS = 0.35
FLANK_DISTANCE_WORSE_LIMIT = -0.05

BOUNDARY_SAFE_MARGIN = 0.45
BOUNDARY_DANGER_MARGIN = 0.20
BOUNDARY_MARGIN_IMPROVE_STRONG = 0.12
BOUNDARY_MARGIN_IMPROVE_MIN = 0.08
BOUNDARY_DISTANCE_WORSE_LIMIT = -0.10

VERTICAL_BAND_LOW = 0.25
VERTICAL_BAND_HIGH = 0.70
VERTICAL_MAINTAIN_DROP_MAX = 0.12
VERTICAL_Z_MARGIN_SAFE = 0.20
VERTICAL_DANGER_MARGIN = 0.15


def run_episode(model: Any, policy: str, scenario: str, skill_horizon: int) -> dict[str, float | int | str | bool]:
    from src.training.sac_env import PursuitEscapeGymEnv

    env = PursuitEscapeGymEnv(scenario=scenario)
    obs, _ = env.reset()
    start = dict(env.inner._observation(closing_speed=0.0))

    start_dist = float(start["distance"])
    start_close = float(start["closing_speed"])
    start_threat_right = abs(float(start["threat_right"]))
    start_margin = float(start["min_boundary_margin"])
    start_z_sep = abs(float(start["dz"]))
    start_evader_z = env.inner.state.evader.z

    completion_step = skill_horizon
    skill_completed = False
    handoff_to_boundary = False
    z_out_of_bounds = False
    out_of_bounds = False
    capture_before_completion = False
    danger_boundary_seen = start_margin <= BOUNDARY_DANGER_MARGIN

    last = start
    outcome = "running"

    for step in range(1, skill_horizon + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        last = env.inner._observation(closing_speed=float(info.get("closing_speed", 0.0)))
        outcome = str(info.get("outcome", "running"))

        dist_gain_now = float(last["distance"]) - start_dist
        close_now = float(last["closing_speed"])
        right_now = abs(float(last["threat_right"]))
        margin_now = float(last["min_boundary_margin"])
        z_sep_now = abs(float(last["dz"]))
        z_margin_now = float(last["boundary_margin_z"])
        danger_boundary_seen = danger_boundary_seen or margin_now <= BOUNDARY_DANGER_MARGIN

        if policy == "pi1":
            skill_completed = (
                (dist_gain_now >= REAR_DIST_GAIN_MIN or float(last["distance"]) >= REAR_SAFE_DISTANCE)
                and close_now <= REAR_CLOSE_RELIEF_MAX
                and outcome != "captured"
            )
        elif policy == "pi2":
            skill_completed = (
                ((start_threat_right - right_now) >= FLANK_THREAT_REDUCTION_MIN or right_now <= FLANK_SAFE_THREAT_RIGHT_ABS)
                and dist_gain_now >= FLANK_DISTANCE_WORSE_LIMIT
                and outcome != "captured"
            )
        elif policy == "pi3":
            strong_return = margin_now >= BOUNDARY_SAFE_MARGIN
            recovered_not_danger = (margin_now - start_margin) >= BOUNDARY_MARGIN_IMPROVE_STRONG and margin_now > BOUNDARY_DANGER_MARGIN
            mild_recovered = (margin_now - start_margin) >= BOUNDARY_MARGIN_IMPROVE_MIN and margin_now > (BOUNDARY_DANGER_MARGIN + 0.08)
            skill_completed = (
                (strong_return or recovered_not_danger or mild_recovered)
                and dist_gain_now >= BOUNDARY_DISTANCE_WORSE_LIMIT
                and outcome != "out_of_bounds"
            )
        else:
            in_target_band = VERTICAL_BAND_LOW <= z_sep_now <= VERTICAL_BAND_HIGH
            started_in_band = VERTICAL_BAND_LOW <= start_z_sep <= VERTICAL_BAND_HIGH
            maintained_band = started_in_band and z_sep_now >= max(VERTICAL_BAND_LOW, start_z_sep - VERTICAL_MAINTAIN_DROP_MAX)
            skill_completed = (
                (in_target_band or maintained_band)
                and z_margin_now >= VERTICAL_Z_MARGIN_SAFE
                and outcome != "captured"
                and not z_out_of_bounds
            )

        if policy in {"pi1", "pi2", "pi4"} and skill_completed and (margin_now <= BOUNDARY_DANGER_MARGIN):
            handoff_to_boundary = True

        if skill_completed:
            completion_step = step
            break

        if outcome == "captured":
            capture_before_completion = True
            break

        if outcome == "out_of_bounds":
            out_of_bounds = True
            s = env.inner.state
            assert s is not None
            z_out_of_bounds = s.evader.z <= env.inner.term_cfg.z_min or s.evader.z >= env.inner.term_cfg.z_max
            if policy in {"pi1", "pi2", "pi4"} and not z_out_of_bounds:
                handoff_to_boundary = True
            break

        if terminated or truncated:
            break

    s_end = env.inner.state
    assert s_end is not None
    end_dist = float(last["distance"])
    end_close = float(last["closing_speed"])
    end_threat_right = abs(float(last["threat_right"]))
    end_margin = float(last["min_boundary_margin"])
    end_z_sep = abs(float(last["dz"]))

    vertical_target_band = VERTICAL_BAND_LOW <= end_z_sep <= VERTICAL_BAND_HIGH
    vertical_maintained = (
        (VERTICAL_BAND_LOW <= start_z_sep <= VERTICAL_BAND_HIGH and end_z_sep >= max(VERTICAL_BAND_LOW, start_z_sep - VERTICAL_MAINTAIN_DROP_MAX))
        or vertical_target_band
    )

    if policy in {"pi1", "pi2", "pi4"} and (danger_boundary_seen or end_margin <= BOUNDARY_DANGER_MARGIN):
        handoff_to_boundary = True

    z_oob_direction = "none"
    if z_out_of_bounds:
        if s_end.evader.z <= env.inner.term_cfg.z_min:
            z_oob_direction = "z_min"
        elif s_end.evader.z >= env.inner.term_cfg.z_max:
            z_oob_direction = "z_max"

    return {
        "outcome": outcome,
        "skill_completed": skill_completed,
        "completion_step": completion_step,
        "distance_gain": end_dist - start_dist,
        "closing_speed_reduction": start_close - end_close,
        "threat_right_abs_reduction": start_threat_right - end_threat_right,
        "lateral_threat_reduction": start_threat_right - end_threat_right,
        "min_boundary_margin_improvement": end_margin - start_margin,
        "return_to_safe_region": end_margin >= BOUNDARY_SAFE_MARGIN,
        "vertical_separation_gain": end_z_sep - start_z_sep,
        "final_vertical_separation": end_z_sep,
        "min_vertical_separation": min(start_z_sep, end_z_sep),
        "max_vertical_separation": max(start_z_sep, end_z_sep),
        "final_z_change": s_end.evader.z - start_evader_z,
        "vertical_target_band": vertical_target_band,
        "vertical_separation_maintenance": vertical_maintained,
        "controlled_z_margin": float(last["boundary_margin_z"]) >= VERTICAL_Z_MARGIN_SAFE,
        "capture_avoidance": not capture_before_completion,
        "out_of_bounds": out_of_bounds or outcome == "out_of_bounds",
        "z_out_of_bounds": z_out_of_bounds,
        "z_oob_direction": z_oob_direction,
        "handoff_to_boundary": handoff_to_boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Option-level low-level skill diagnostics on home scenarios")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--skill-horizon", type=int, default=80)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    checkpoint_dir = Path(args.checkpoint_dir)
    for filename in MODEL_FILENAMES:
        path = checkpoint_dir / filename
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    rows: list[dict[str, Any]] = []
    for policy, scenario, filename in SKILL_ROWS:
        model = SAC.load(str(checkpoint_dir / filename))
        eps = [run_episode(model, policy, scenario, args.skill_horizon) for _ in range(args.episodes)]
        n = max(len(eps), 1)

        def rate(key: str) -> float:
            return sum(1 for e in eps if bool(e[key])) / n

        def avg(key: str) -> float:
            return sum(float(e[key]) for e in eps) / n

        z_oob_min_rate = sum(1 for e in eps if e["z_oob_direction"] == "z_min") / n
        z_oob_max_rate = sum(1 for e in eps if e["z_oob_direction"] == "z_max") / n
        skill_success_rate = rate("skill_completed")
        row = {
            "policy": policy,
            "scenario": scenario,
            "episodes": args.episodes,
            "skill_horizon": args.skill_horizon,
            "skill_success_rate": skill_success_rate,
            "skill_completed_rate": rate("skill_completed"),
            "avg_completion_step": avg("completion_step"),
            "distance_gain": avg("distance_gain"),
            "closing_speed_reduction": avg("closing_speed_reduction"),
            "threat_right_abs_reduction": avg("threat_right_abs_reduction"),
            "min_boundary_margin_improvement": avg("min_boundary_margin_improvement"),
            "return_to_safe_region_rate": rate("return_to_safe_region"),
            "vertical_separation_gain": avg("vertical_separation_gain"),
            "final_vertical_separation": avg("final_vertical_separation"),
            "min_vertical_separation": avg("min_vertical_separation"),
            "max_vertical_separation": avg("max_vertical_separation"),
            "final_z_change": avg("final_z_change"),
            "vertical_target_band_rate": rate("vertical_target_band"),
            "vertical_separation_maintenance_rate": rate("vertical_separation_maintenance"),
            "controlled_z_margin_rate": rate("controlled_z_margin"),
            "out_of_bounds_rate": rate("out_of_bounds"),
            "z_out_of_bounds_rate": rate("z_out_of_bounds"),
            "z_oob_min_rate": z_oob_min_rate,
            "z_oob_max_rate": z_oob_max_rate,
            "handoff_to_boundary_rate": rate("handoff_to_boundary"),
        }
        rows.append(row)
        print(
            f"{policy}@{scenario}: skill_success_rate={skill_success_rate:.3f}, "
            f"return_to_safe={row['return_to_safe_region_rate']:.3f}, "
            f"handoff={row['handoff_to_boundary_rate']:.3f}, z_oob={row['z_out_of_bounds_rate']:.3f}"
        )

    out_csv = Path("outputs/evaluation/lowlevel_skill_diagnostics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] saved: {out_csv}")


if __name__ == "__main__":
    main()
