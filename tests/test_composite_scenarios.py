from src.env.dynamics import Env3DState
from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.training.highlevel_env import COMPOSITE_SCENARIOS


def composite_observation(name: str) -> dict[str, float]:
    env = PursuitEscapeEnv()
    evader, pursuer = COMPOSITE_SCENARIOS[name]
    env.state = Env3DState(evader=evader, pursuer=pursuer, step_count=0)
    return env._observation(closing_speed=0.0)


def test_flank_boundary_composite_has_recoverable_flank_pressure() -> None:
    obs = composite_observation("composite_flank_boundary")

    assert 0.14 <= obs["min_boundary_margin"] <= 0.20
    assert abs(obs["threat_right"]) >= 0.65
    assert obs["threat_forward"] <= 0.10


def test_rear_flank_boundary_composite_combines_all_three_pressures() -> None:
    obs = composite_observation("composite_rear_flank_boundary")

    assert 0.14 <= obs["min_boundary_margin"] <= 0.20
    assert obs["threat_forward"] <= -0.35
    assert abs(obs["threat_right"]) >= 0.55
