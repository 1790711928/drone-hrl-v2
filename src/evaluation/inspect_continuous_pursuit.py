from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.training.highlevel_env import HighLevelOptionEnv


class ZeroModel:
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        return np.zeros(3, dtype=np.float32), None


FIELDS = (
    "threat_forward",
    "threat_right",
    "threat_up",
    "min_boundary_margin",
    "distance",
    "closing_speed",
    "evader_x_norm",
    "evader_y_norm",
    "evader_z_norm",
)


def observation_row(env: HighLevelOptionEnv, info: dict[str, Any]) -> dict[str, float | str | int]:
    obs = env.inner.inner._observation(float(info.get("closing_speed", 0.0)))
    scores = dict(info.get("threat_scores", env._continuous_threat_scores(obs, float(info.get("closing_speed", 0.0)))))
    row: dict[str, float | str | int | bool] = {
        "scheduled_regime": str(info.get("scheduled_regime", env._continuous_scheduled_regime())),
        "regime_name": str(info.get("regime_name", env._continuous_regime())),
        "lowlevel_step": int(info.get("continuous_lowlevel_steps", env.continuous_lowlevel_steps)),
        "boundary_priority_active": bool(info.get("boundary_priority_active", False)),
        "state_driven_regime_active": bool(info.get("state_driven_regime_active", False)),
        "selected_option": str(info.get("selected_option", "none")),
        "outcome": str(info.get("outcome", "running")),
        "score_boundary": float(scores.get("boundary", 0.0)),
        "score_rear": float(scores.get("rear", 0.0)),
        "score_flank": float(scores.get("flank", 0.0)),
        "score_vertical": float(scores.get("vertical", 0.0)),
    }
    for field in FIELDS:
        row[field] = float(obs[field])
    return row


def print_row(row: dict[str, float | str | int | bool]) -> None:
    print(
        "step={lowlevel_step:>4} scheduled={scheduled_regime:<8} actual={regime_name:<8} option={selected_option} "
        "boundary_priority={boundary_priority_active} state_driven={state_driven_regime_active} "
        "scores[b={score_boundary:.2f},r={score_rear:.2f},f={score_flank:.2f},v={score_vertical:.2f}] "
        "tf={threat_forward:+.3f} tr={threat_right:+.3f} tu={threat_up:+.3f} "
        "margin={min_boundary_margin:.3f} dist={distance:.3f} closing={closing_speed:+.3f} "
        "ex={evader_x_norm:+.3f} ey={evader_y_norm:+.3f} ez={evader_z_norm:+.3f} "
        "outcome={outcome}".format(**row)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect regime geometry in the continuous_pursuit benchmark")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-lowlevel-steps", type=int, default=400)
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    parser.add_argument("--min-regime-hold-steps", type=int, default=20)
    parser.add_argument("--boundary-priority-enter", type=float, default=0.24)
    parser.add_argument("--boundary-priority-exit", type=float, default=0.32)
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.20)
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        option_duration=args.option_duration,
        scenario_set="continuous_pursuit",
        episode_lowlevel_steps=args.episode_lowlevel_steps,
        regime_duration=args.regime_duration,
        regime_schedule=args.regime_schedule,
        min_regime_hold_steps=args.min_regime_hold_steps,
        boundary_priority_enter=args.boundary_priority_enter,
        boundary_priority_exit=args.boundary_priority_exit,
        pursuer_speed_ratio=args.pursuer_speed_ratio,
    )

    print("=== continuous_pursuit regime geometry ===")
    print(
        f"episodes={args.episodes}, episode_lowlevel_steps={args.episode_lowlevel_steps}, "
        f"regime_duration={args.regime_duration}, schedule={args.regime_schedule}"
    )
    for episode in range(args.episodes):
        _, info = env.reset(options={"scenario_set": "continuous_pursuit"})
        print(f"\nEpisode {episode + 1}")
        initial_row = observation_row(env, info)
        last_marker = (str(initial_row["regime_name"]), bool(initial_row["boundary_priority_active"]))
        last_print_step = int(initial_row["lowlevel_step"])
        print_row(initial_row)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            row = observation_row(env, info)
            marker = (str(row["regime_name"]), bool(row["boundary_priority_active"]))
            step = int(row["lowlevel_step"])
            should_print_interval = args.print_every > 0 and step - last_print_step >= args.print_every
            if marker != last_marker or should_print_interval:
                print_row(row)
                last_marker = marker
                last_print_step = step
            done = bool(terminated or truncated)
        print(
            "outcome={outcome}, lowlevel_steps={continuous_lowlevel_steps}, "
            "recent_distance={recent_distance:.3f}, recent_closing_speed={recent_closing_speed:+.3f}, "
            "regime_coverage_rate={regime_coverage_rate:.3f}, boundary_priority_rate={boundary_priority_rate:.3f}, "
            "state_driven_switches={state_driven_regime_switch_count}".format(**info)
        )


if __name__ == "__main__":
    main()
