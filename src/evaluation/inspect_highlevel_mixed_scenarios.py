from __future__ import annotations

from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.training.highlevel_env import MIXED_SCENARIOS


def main() -> None:
    env = PursuitEscapeEnv()
    print("=== High-level mixed/composite scenario sanity ===")
    for mixed_name, mix in MIXED_SCENARIOS.items():
        print(f"\n[{mixed_name}] composition={mix}")
        for base in mix.keys():
            obs = env.reset(scenario=base, randomize=False)
            print(
                f"  - {base:<22} threat_f={obs['threat_forward']:+.3f} "
                f"threat_r={obs['threat_right']:+.3f} threat_u={obs['threat_up']:+.3f} "
                f"min_margin={obs['min_boundary_margin']:.3f} dist={obs['distance']:.3f} los={obs['los_cos']:+.3f}"
            )


if __name__ == "__main__":
    main()
