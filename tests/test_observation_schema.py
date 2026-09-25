import numpy as np

from src.training.sac_env import PursuitEscapeGymEnv


def test_sac_env_observation_shape_matches_schema_on_reset_and_step() -> None:
    env = PursuitEscapeGymEnv(scenario="rear_close_threat", randomize_reset=False)
    obs, _ = env.reset()
    expected_dim = len(PursuitEscapeGymEnv.OBS_KEYS)

    assert obs.shape == (expected_dim,)
    assert env.observation_space.shape == (expected_dim,)

    next_obs, _, _, _, _ = env.step(np.array([0.1, 0.0, 0.0], dtype=np.float32))
    assert next_obs.shape == (expected_dim,)
