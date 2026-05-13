from __future__ import annotations

from src.env.pursuit_escape_env import PursuitEscapeEnv

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]

KEYS = [
    "dx",
    "dy",
    "dz",
    "evader_pitch",
    "pursuer_pitch",
    "evader_x_norm",
    "evader_y_norm",
    "evader_z_norm",
    "threat_forward",
    "threat_right",
    "threat_up",
    "distance",
    "closing_speed",
    "los_cos",
    "boundary_margin_x",
    "boundary_margin_y",
    "boundary_margin_z",
    "min_boundary_margin",
]


def main() -> None:
    env = PursuitEscapeEnv()
    print("=== Observation Sanity Check by Scenario ===")
    print("fields:", ", ".join(KEYS))

    for scenario in SCENARIOS:
        obs = env.reset(scenario=scenario, randomize=False)
        values = " | ".join([f"{k}={obs[k]: .4f}" for k in KEYS])
        print(f"{scenario:>22} -> {values}")


if __name__ == "__main__":
    main()
