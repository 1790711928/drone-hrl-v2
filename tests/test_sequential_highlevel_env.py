from dataclasses import replace

import numpy as np

from src.env.dynamics import Env3DState
from src.training.highlevel_env import HighLevelOptionEnv, SEQUENTIAL_SCENARIOS, inject_sequential_phase


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def test_sequential_phase_injection_exposes_distinct_geometry() -> None:
    evader = SEQUENTIAL_SCENARIOS["sequential_rear_to_flank_to_boundary"].evader
    rear = inject_sequential_phase(evader, "rear")
    flank = inject_sequential_phase(evader, "flank")
    boundary = inject_sequential_phase(evader, "boundary")

    env = HighLevelOptionEnv(low_models=[ZeroModel()] * 4, scenario_set="sequential")
    env.inner.inner.state = rear
    rear_obs = env.inner.inner._observation(0.0)
    env.inner.inner.state = flank
    flank_obs = env.inner.inner._observation(0.0)
    env.inner.inner.state = boundary
    boundary_obs = env.inner.inner._observation(0.0)

    assert rear_obs["threat_forward"] <= -0.95
    assert abs(flank_obs["threat_right"]) >= 0.90
    assert boundary_obs["min_boundary_margin"] <= 0.20


def test_sequential_env_advances_phase_without_finishing_episode() -> None:
    env = HighLevelOptionEnv(low_models=[ZeroModel()] * 4, option_duration=1, scenario_set="sequential")
    _, info = env.reset(options={"scenario_set": "sequential", "scenario_name": "sequential_rear_to_boundary"})
    assert info["phase_name"] == "rear"

    state = env.inner.inner.state
    assert state is not None
    env.inner.inner.state = Env3DState(evader=state.evader, pursuer=replace(state.pursuer, x=-40.0), step_count=0)

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert info["phase_transition"] is True
    assert info["completed_phases"] == 1
    assert info["phase_name"] == "boundary"
