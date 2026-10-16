import numpy as np

from src.training.highlevel_env import HighLevelOptionEnv, STRUCTURED_RANDOM_COMPOSITE_SCENARIO


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def _make_env(**kwargs):
    return HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set=STRUCTURED_RANDOM_COMPOSITE_SCENARIO,
        option_duration=1,
        episode_lowlevel_steps=8,
        regime_duration=2,
        **kwargs,
    )


def test_structured_random_composite_same_seed_reproduces_state_and_schedule():
    first = _make_env()
    second = _make_env()
    _, first_info = first.reset(seed=17, options={"scenario_set": STRUCTURED_RANDOM_COMPOSITE_SCENARIO})
    _, second_info = second.reset(seed=17, options={"scenario_set": STRUCTURED_RANDOM_COMPOSITE_SCENARIO})

    assert first.inner.inner.state == second.inner.inner.state
    assert first_info["structured_template"] == second_info["structured_template"]
    assert first_info["regime_schedule"] == second_info["regime_schedule"]


def test_structured_random_composite_jitter_is_configurable():
    env = _make_env(
        structured_position_jitter=0.0,
        structured_relative_jitter=0.0,
        structured_yaw_jitter=0.0,
        structured_pitch_jitter=0.0,
        structured_speed_jitter=0.0,
    )
    _, info = env.reset(seed=3, options={"scenario_set": STRUCTURED_RANDOM_COMPOSITE_SCENARIO})
    assert info["scenario_set"] == STRUCTURED_RANDOM_COMPOSITE_SCENARIO
    assert info["structured_template"].startswith("composite_")
    assert env.inner.inner.state is not None


def test_structured_random_composite_step_keeps_observation_schema():
    env = _make_env()
    obs, _ = env.reset(seed=4, options={"scenario_set": STRUCTURED_RANDOM_COMPOSITE_SCENARIO})
    next_obs, _, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert next_obs.shape == env.observation_space.shape
    assert info["scenario_set"] == STRUCTURED_RANDOM_COMPOSITE_SCENARIO
    assert not (terminated and truncated)
