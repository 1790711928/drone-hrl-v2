from __future__ import annotations

from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.env.reward import compute_reward_terms
from src.env.termination import EpisodeOutcome

SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]


def main() -> None:
    env = PursuitEscapeEnv()
    print("=== Reward Term Sanity Check by Scenario ===")
    print("note: prints single-step example terms after reset with demo action (0.2, 0.05, 0.01).")
    for scenario in SCENARIOS:
        env.reset(scenario=scenario, randomize=False)
        assert env.state is not None
        prev_distance = env.inner_distance if hasattr(env, "inner_distance") else None
        # derive prev/cur by running exactly one environment step with fixed action
        pre_state = env.state
        prev_d = ((pre_state.evader.x - pre_state.pursuer.x) ** 2 + (pre_state.evader.y - pre_state.pursuer.y) ** 2 + (pre_state.evader.z - pre_state.pursuer.z) ** 2) ** 0.5
        obs, reward, done, info = env.step((0.2, 0.05, 0.01))
        post_state = env.state
        assert post_state is not None
        cur_d = ((post_state.evader.x - post_state.pursuer.x) ** 2 + (post_state.evader.y - post_state.pursuer.y) ** 2 + (post_state.evader.z - post_state.pursuer.z) ** 2) ** 0.5
        terms = compute_reward_terms(
            scenario=scenario,
            prev_distance=prev_d,
            cur_distance=cur_d,
            action=(0.2, 0.05, 0.01),
            evader_position=(post_state.evader.x, post_state.evader.y, post_state.evader.z),
            pursuer_position=(post_state.pursuer.x, post_state.pursuer.y, post_state.pursuer.z),
            bounds=(
                env.term_cfg.x_min,
                env.term_cfg.x_max,
                env.term_cfg.y_min,
                env.term_cfg.y_max,
                env.term_cfg.z_min,
                env.term_cfg.z_max,
            ),
            outcome=EpisodeOutcome(info["outcome"]),
        )
        print(
            f"{scenario:>22} -> distance={terms['distance_term']:.4f}, boundary={terms['boundary_term']:.4f}, "
            f"scenario={terms['scenario_term']:.4f}, total={terms['total_reward']:.4f}"
        )


if __name__ == "__main__":
    main()
