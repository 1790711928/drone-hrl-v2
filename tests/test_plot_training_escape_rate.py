import csv

import pytest

from src.evaluation.plot_training_escape_rate import _read_escape_rows


def test_read_escape_rows_uses_global_step_and_rolling_window(tmp_path):
    path = tmp_path / "episode_rewards.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "global_step", "escaped"])
        writer.writeheader()
        writer.writerows(
            [
                {"episode_index": 1, "global_step": 10, "escaped": 0},
                {"episode_index": 2, "global_step": 20, "escaped": 1},
                {"episode_index": 3, "global_step": 30, "escaped": 1},
            ]
        )

    rows = _read_escape_rows(path, window=2)
    assert [row["global_step"] for row in rows] == [10, 20, 30]
    assert [row["escape_rate"] for row in rows] == [0.0, 0.5, 1.0]


def test_read_escape_rows_rejects_old_reward_only_logs(tmp_path):
    path = tmp_path / "episode_rewards.csv"
    path.write_text("global_step,episode_return\n10,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outcome logging"):
        _read_escape_rows(path, window=20)
