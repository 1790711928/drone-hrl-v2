from src.env.pursuit_escape_env import PursuitEscapeEnv


def test_env_reset_and_step_returns_expected_fields() -> None:
    env = PursuitEscapeEnv()
    obs = env.reset("s1_close_threat")
    expected_keys = {
        "dx",
        "dy",
        "dz",
        "distance",
        "closing_speed",
        "evader_speed",
        "pursuer_speed",
        "evader_yaw_sin",
        "evader_yaw_cos",
        "pursuer_yaw_sin",
        "pursuer_yaw_cos",
        "evader_pitch",
        "pursuer_pitch",
        "los_cos",
        "boundary_margin_x",
        "boundary_margin_y",
        "boundary_margin_z",
        "min_boundary_margin",
        "normalized_step",
        "evader_x_norm",
        "evader_y_norm",
        "evader_z_norm",
        "threat_forward",
        "threat_right",
        "threat_up",
    }
    assert set(obs.keys()) == expected_keys

    next_obs, reward, done, info = env.step((0.2, 0.01, 0.0))
    assert set(next_obs.keys()) == expected_keys
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
