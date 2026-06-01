import numpy as np

from src.evaluation.plot_highlevel_trajectories import (
    EpisodePlotData,
    TrajectoryRecorder,
    compressed_option_sequence,
    rollout_episode,
    select_episodes_for_plot,
)
from src.training.highlevel_env import HighLevelOptionEnv


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def episode(*, mode="fixed", success=False, scenario="sequential_rear_to_boundary", episode_id=1):
    return EpisodePlotData(
        scenario=scenario,
        mode=mode,
        episode_id=episode_id,
        outcome="escaped" if success else "timeout",
        success=success,
        switch_count=0,
        option_sequence=[2],
        completed_phases=1,
        evader_points=[(0.0, 0.0, 0.0)],
        pursuer_points=[(1.0, 0.0, 0.0)],
        option_switches=[(0, "pi3")],
        phase_starts=[(0, "rear")],
    )


def test_compressed_option_sequence_removes_repeated_selections():
    assert compressed_option_sequence([2, 2, 1, 1, 3]) == [2, 1, 3]


def test_plot_selection_prioritizes_highlevel_success_then_fixed_pi3_failure():
    candidates = [
        episode(mode="fixed", success=True, episode_id=1),
        episode(mode="fixed", success=False, episode_id=2),
        episode(mode="highlevel", success=True, episode_id=3),
    ]
    selected = select_episodes_for_plot(candidates, max_plots=2, fixed_policy=2, only_success=False, only_failure=False)
    assert [item.episode_id for item in selected] == [3, 2]


def test_rollout_recorder_collects_lowlevel_points_and_summary():
    env = HighLevelOptionEnv([ZeroModel()] * 4, option_duration=1, max_highlevel_steps=1, scenario_set="sequential")
    recorder = TrajectoryRecorder(env)
    recorder.attach()
    result = rollout_episode(
        env,
        recorder,
        episode_id=7,
        mode="fixed",
        fixed_policy=2,
        high_model=None,
        scenario_set="sequential",
    )
    assert len(result.evader_points) >= 2
    assert len(result.evader_points) == len(result.pursuer_points)
    assert result.option_sequence == [2]
    assert result.option_switches == [(0, "pi3")]
    assert result.summary_row()["episode_id"] == 7
