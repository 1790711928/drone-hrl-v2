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
    assert info["regime_name"] == "rear"

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
    assert len(set(regimes)) >= 2
