from src.env.pursuit_escape_env import PursuitEscapeEnv


def test_env_reset_and_step_returns_expected_fields() -> None:
    env = PursuitEscapeEnv()
    obs = env.reset("s1_close_threat")
    assert "distance" in obs

    next_obs, reward, done, info = env.step((0.2, 0.01, 0.0))
    assert "distance" in next_obs
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "outcome" in info


def test_step_before_reset_raises() -> None:
    env = PursuitEscapeEnv()
    try:
        env.step((0.0, 0.0, 0.0))
    except RuntimeError:
        return
    raise AssertionError("step() before reset() should raise RuntimeError")
