import numpy as np

from src.training.highlevel_env import HighLevelOptionEnv


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def test_continuous_pursuit_progresses_without_phase_completion():
    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set="continuous_pursuit",
        option_duration=4,
        episode_lowlevel_steps=20,
        regime_duration=5,
    )
    obs, info = env.reset(options={"scenario_set": "continuous_pursuit"})
    assert info["scenario_set"] == "continuous_pursuit"
    assert info["regime_name"] in {"rear", "boundary"}
    assert "boundary_priority_active" in info

    obs, reward, terminated, truncated, info = env.step(2)
    assert obs.shape == env.observation_space.shape
    assert info["completed_phases"] == 0
    assert info["total_phases"] == 0
    assert info["continuous_lowlevel_steps"] == 4
    assert info["regime_name"] in {"rear", "vertical", "boundary", "flank"}
    assert not (terminated and truncated)


def test_continuous_pursuit_regime_changes_do_not_reset_evader():
    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set="continuous_pursuit",
        option_duration=1,
        episode_lowlevel_steps=12,
        regime_duration=2,
    )
    env.reset(options={"scenario_set": "continuous_pursuit"})
    previous = env.inner.inner.state.evader
    regimes = []
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(0)
        current = env.inner.inner.state.evader
        regimes.append(info["regime_name"])
        jump = ((current.x - previous.x) ** 2 + (current.y - previous.y) ** 2 + (current.z - previous.z) ** 2) ** 0.5
        assert jump < 5.0
        previous = current
        if terminated or truncated:
            break
    assert regimes


def test_continuous_showcase_uses_scaled_bounds_and_continuous_state():
    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set="continuous_showcase",
        option_duration=2,
        episode_lowlevel_steps=12,
        showcase_bound_scale=2.5,
        showcase_z_bound_scale=1.5,
    )
    obs, info = env.reset(options={"scenario_set": "continuous_showcase"})
    assert info["scenario_set"] == "continuous_showcase"
    assert env.inner.inner.term_cfg.x_max == 125.0
    assert env.inner.inner.term_cfg.y_max == 125.0
    assert env.inner.inner.term_cfg.z_max == 75.0
    start_evader = env.inner.inner.state.evader

    obs, _, terminated, truncated, info = env.step(0)
    current_evader = env.inner.inner.state.evader
    jump = ((current_evader.x - start_evader.x) ** 2 + (current_evader.y - start_evader.y) ** 2 + (current_evader.z - start_evader.z) ** 2) ** 0.5
    assert obs.shape == env.observation_space.shape
    assert jump < 5.0
    assert info["completed_phases"] == 0
    assert not (terminated and truncated)


def test_continuous_showcase_bounds_do_not_leak_to_sequential_reset():
    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set="continuous_showcase",
    )
    env.reset(options={"scenario_set": "continuous_showcase"})
    assert env.inner.inner.term_cfg.x_max == 125.0
    env.reset(options={"scenario_set": "sequential"})
    assert env.inner.inner.term_cfg.x_max == 50.0


def test_scripted_showcase_uses_scripted_regime_and_expanded_bounds():
    env = HighLevelOptionEnv(
        low_models=[ZeroModel(), ZeroModel(), ZeroModel(), ZeroModel()],
        scenario_set="scripted_showcase",
        option_duration=1,
        episode_lowlevel_steps=170,
    )
    _, info = env.reset(options={"scenario_set": "scripted_showcase"})
    assert info["scenario_set"] == "scripted_showcase"
    assert info["regime_name"] == "rear"
    assert env.inner.inner.term_cfg.x_max == 200.0
    assert env.inner.inner.term_cfg.y_max == 200.0
    assert env.inner.inner.term_cfg.z_max == 100.0
    start_evader = env.inner.inner.state.evader

    regimes = []
    for _ in range(170):
        _, _, terminated, truncated, info = env.step({"rear": 0, "flank": 1, "vertical": 3, "boundary": 2}.get(info["regime_name"], 0))
        regimes.append(info["regime_name"])
        if terminated or truncated:
            break
    current_evader = env.inner.inner.state.evader
    jump_from_start = ((current_evader.x - start_evader.x) ** 2 + (current_evader.y - start_evader.y) ** 2 + (current_evader.z - start_evader.z) ** 2) ** 0.5
    assert jump_from_start > 0.0
    assert {"rear", "flank", "vertical"}.issubset(set(regimes))
    assert info["completed_phases"] == 0
