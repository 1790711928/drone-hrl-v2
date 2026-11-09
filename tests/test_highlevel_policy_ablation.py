from src.evaluation.evaluate_highlevel_checkpoints import _episode_disabled_policy


def test_balanced_random_disable_covers_each_policy_once_per_block():
    disabled = [_episode_disabled_policy("balanced_random", 1.0, 123, trial) for trial in range(8)]
    assert sorted(disabled[:4]) == [0, 1, 2, 3]
    assert sorted(disabled[4:]) == [0, 1, 2, 3]


def test_disable_probability_zero_keeps_all_policies_available():
    assert all(
        _episode_disabled_policy("balanced_random", 0.0, 123, trial) is None
        for trial in range(20)
    )


def test_none_mode_does_not_disable_a_policy():
    assert _episode_disabled_policy("none", 1.0, 123, 0) is None
