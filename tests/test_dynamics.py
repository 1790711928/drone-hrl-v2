from src.env.dynamics import Agent3DState, rule_based_pursuer_control, step_kinematics


def test_rule_based_pursuer_returns_controls() -> None:
    e = Agent3DState(x=10, y=0, z=10, speed=10, yaw=0, pitch=0)
    p = Agent3DState(x=0, y=0, z=0, speed=8, yaw=0, pitch=0)
    accel, yaw_rate, pitch_rate = rule_based_pursuer_control(e, p, pursuer_speed_ratio=1.1)
    assert accel > 0
    assert isinstance(yaw_rate, float)
    assert isinstance(pitch_rate, float)


def test_step_kinematics_updates_position() -> None:
    s = Agent3DState(x=0, y=0, z=10, speed=5, yaw=0, pitch=0)
    next_s = step_kinematics(s, accel=1.0, yaw_rate=0.1, pitch_rate=0.0, dt=0.1,
                             speed_min=1.0, speed_max=20.0, yaw_rate_max=1.0, pitch_rate_max=1.0)
    assert next_s.x > s.x
    assert next_s.z >= 10
