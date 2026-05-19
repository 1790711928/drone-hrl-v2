from __future__ import annotations

import argparse

from src.env.dynamics import Env3DState
from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.training.highlevel_env import BASIC_SCENARIOS, COMPOSITE_SCENARIOS, MIXED_SCENARIOS


def print_obs_line(name: str, obs: dict[str, float]) -> None:
    print(
        f"  - {name:<28} threat_f={obs['threat_forward']:+.3f} "
        f"threat_r={obs['threat_right']:+.3f} threat_u={obs['threat_up']:+.3f} "
        f"min_margin={obs['min_boundary_margin']:.3f} dist={obs['distance']:.3f} los={obs['los_cos']:+.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect high-level scenario geometry")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite"], default="composite")
    args = parser.parse_args()

    env = PursuitEscapeEnv()
    print(f"=== High-level scenario sanity ({args.scenario_set}) ===")

    if args.scenario_set == "basic":
        for name in BASIC_SCENARIOS:
            obs = env.reset(scenario=name, randomize=False)
            print_obs_line(name, obs)
        return

    if args.scenario_set == "mixed":
        for mixed_name, mix in MIXED_SCENARIOS.items():
            print(f"\n[{mixed_name}] composition={mix}")
            for base in mix.keys():
                obs = env.reset(scenario=base, randomize=False)
                print_obs_line(base, obs)
        return

    for name, (ev, pu) in COMPOSITE_SCENARIOS.items():
        env.state = Env3DState(evader=ev, pursuer=pu, step_count=0)
        obs = env._observation(closing_speed=0.0)
        print_obs_line(name, obs)


if __name__ == "__main__":
    main()
