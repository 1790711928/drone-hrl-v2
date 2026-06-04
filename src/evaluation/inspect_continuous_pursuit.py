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
    row: dict[str, float | str | int] = {
        "regime_name": str(info.get("regime_name", env._continuous_regime())),
        "lowlevel_step": int(info.get("continuous_lowlevel_steps", env.continuous_lowlevel_steps)),
    }
    for field in FIELDS:
        row[field] = float(obs[field])
    return row


def print_row(row: dict[str, float | str | int]) -> None:
    print(
        "regime={regime_name:<8} step={lowlevel_step:>4} "
        "tf={threat_forward:+.3f} tr={threat_right:+.3f} tu={threat_up:+.3f} "
        "margin={min_boundary_margin:.3f} dist={distance:.3f} closing={closing_speed:+.3f} "
        "ex={evader_x_norm:+.3f} ey={evader_y_norm:+.3f} ez={evader_z_norm:+.3f}".format(**row)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect regime geometry in the continuous_pursuit benchmark")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-lowlevel-steps", type=int, default=400)
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.25)
    parser.add_argument("--option-duration", type=int, default=8)
    args = parser.parse_args()

    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        option_duration=args.option_duration,
        scenario_set="continuous_pursuit",
        episode_lowlevel_steps=args.episode_lowlevel_steps,
        regime_duration=args.regime_duration,
        regime_schedule=args.regime_schedule,
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
        last_regime: str | None = str(initial_row["regime_name"])
        print_row(initial_row)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            regime = str(info.get("regime_name", "unknown"))
            if regime != last_regime:
                print_row(observation_row(env, info))
                last_regime = regime
            done = bool(terminated or truncated)
        print(
            "outcome={outcome}, lowlevel_steps={continuous_lowlevel_steps}, "
            "recent_distance={recent_distance:.3f}, recent_closing_speed={recent_closing_speed:+.3f}, "
            "regime_coverage_rate={regime_coverage_rate:.3f}".format(**info)
        )


if __name__ == "__main__":
    main()
