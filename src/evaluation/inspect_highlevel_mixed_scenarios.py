from __future__ import annotations

import argparse

from src.env.dynamics import Env3DState
from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.training.highlevel_env import (
    BASIC_SCENARIOS,
    COMPOSITE_SCENARIOS,
    MIXED_SCENARIOS,
    SEQUENTIAL_SCENARIOS,
    inject_sequential_phase,
)


def print_obs_line(name: str, obs: dict[str, float]) -> None:
    print(
        f"  - {name:<28} ex={obs['evader_x_norm']:+.3f} ey={obs['evader_y_norm']:+.3f} ez={obs['evader_z_norm']:+.3f} "
        f"bx={obs['boundary_margin_x']:.3f} by={obs['boundary_margin_y']:.3f} bz={obs['boundary_margin_z']:.3f} "
        f"min_margin={obs['min_boundary_margin']:.3f} tf={obs['threat_forward']:+.3f} tr={obs['threat_right']:+.3f} "
        f"tu={obs['threat_up']:+.3f} dist={obs['distance']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect high-level scenario geometry")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential"], default="composite")
    args = parser.parse_args()

    env = PursuitEscapeEnv()
    print(f"=== High-level scenario sanity ({args.scenario_set}) ===")

    if args.scenario_set == "basic":
        for name in BASIC_SCENARIOS:
            print_obs_line(name, env.reset(scenario=name, randomize=False))
        return

    if args.scenario_set == "mixed":
        for mixed_name, mix in MIXED_SCENARIOS.items():
            print(f"\n[{mixed_name}] composition={mix}")
            for base in mix.keys():
                print_obs_line(base, env.reset(scenario=base, randomize=False))
        return

    if args.scenario_set == "composite":
        for name, (ev, pu) in COMPOSITE_SCENARIOS.items():
            env.state = Env3DState(evader=ev, pursuer=pu, step_count=0)
            print_obs_line(name, env._observation(closing_speed=0.0))
        return

    for name, spec in SEQUENTIAL_SCENARIOS.items():
        print(f"\n[{name}] phases={list(spec.phases)}")
        evader = spec.evader
        for index, phase in enumerate(spec.phases):
            env.state = inject_sequential_phase(evader, phase)
            print_obs_line(f"phase[{index}]={phase}", env._observation(closing_speed=0.0))
            evader = env.state.evader


if __name__ == "__main__":
    main()
