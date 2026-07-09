import numpy as np

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
    assert abs(flank_obs["threat_right"]) >= 0.70
    assert boundary_obs["min_boundary_margin"] <= 0.20


def test_base_escape_requires_phase_specific_success_streak() -> None:
    env = HighLevelOptionEnv(low_models=[ZeroModel()] * 4, option_duration=3, scenario_set="sequential")
    obs, _ = env.reset(options={"scenario_set": "sequential", "scenario_name": "sequential_rear_to_boundary"})
    env._phase_condition = lambda _: True
    env.inner.step = lambda _: (obs, 50.0, True, False, {"outcome": "escaped", "closing_speed": 0.0})

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert info["phase_transition"] is True
    assert info["completed_phases"] == 1
    assert info["phase_name"] == "boundary"
    assert info["phase_success_by_phase_type"] == {"rear": 1}


def test_phase_timeout_records_failure() -> None:
    env = HighLevelOptionEnv(low_models=[ZeroModel()] * 4, option_duration=2, scenario_set="sequential")
    obs, _ = env.reset(options={"scenario_set": "sequential", "scenario_name": "sequential_rear_to_boundary"})
    env.max_phase_lowlevel_steps = 2
    env._phase_condition = lambda _: False
    env.inner.step = lambda _: (obs, 0.0, False, False, {"outcome": "running", "closing_speed": 0.0})

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert info["outcome"] == "timeout"
    assert info["phase_failed"] is True
    assert info["phase_failure_by_phase_type"] == {"rear": 1}


def test_base_escape_does_not_bypass_phase_specific_condition() -> None:
    env = HighLevelOptionEnv(low_models=[ZeroModel()] * 4, option_duration=1, scenario_set="sequential")
    obs, _ = env.reset(options={"scenario_set": "sequential", "scenario_name": "sequential_rear_to_boundary"})
    env._phase_condition = lambda _: False
    env.inner.step = lambda _: (obs, 50.0, True, False, {"outcome": "escaped", "closing_speed": 0.0})

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert info["phase_transition"] is False
    assert info["completed_phases"] == 0
    assert info["phase_name"] == "rear"
