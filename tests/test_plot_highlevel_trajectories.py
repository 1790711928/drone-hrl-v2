import numpy as np

from src.evaluation.plot_highlevel_trajectories import (
    EpisodePlotData,
    TrajectoryRecorder,
    compressed_option_sequence,
    phase_reset_jumps,
    rollout_episode,
    select_episodes_for_plot,
    trajectory_segments,
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
        regime_starts=[],
        lowlevel_steps=0,
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


def test_trajectory_segments_break_at_phase_starts():
    segments = trajectory_segments([(0, "rear"), (3, "boundary")], point_count=6)
    assert segments == [(0, 2, "rear"), (3, 5, "boundary")]
    assert phase_reset_jumps(segments) == [(2, 3)]


def test_selection_can_deduplicate_by_scenario_and_option_sequence():
    candidates = [
        episode(scenario="a", episode_id=1),
        episode(scenario="a", episode_id=2),
        episode(scenario="b", episode_id=3),
    ]
    by_scenario = select_episodes_for_plot(
        candidates, max_plots=3, fixed_policy=2, only_success=False, only_failure=False, one_per_scenario=True
    )
    assert [item.episode_id for item in by_scenario] == [1, 3]

    by_sequence = select_episodes_for_plot(
        candidates, max_plots=3, fixed_policy=2, only_success=False, only_failure=False, one_per_option_sequence=True
    )
    assert [item.episode_id for item in by_sequence] == [1]


def test_rollout_accepts_specific_scenario_name():
    env = HighLevelOptionEnv([ZeroModel()] * 4, option_duration=1, max_highlevel_steps=1, scenario_set="sequential")
    recorder = TrajectoryRecorder(env)
    recorder.attach()
    result = rollout_episode(
        env,
        recorder,
        episode_id=8,
        mode="fixed",
        fixed_policy=2,
        high_model=None,
        scenario_set="sequential",
        scenario_name="sequential_rear_vertical_to_boundary",
    )
    assert result.scenario == "sequential_rear_vertical_to_boundary"
    assert result.phase_starts[0][1] == "rear_vertical"


def test_continuous_rollout_records_regime_switches():
    env = HighLevelOptionEnv(
        [ZeroModel()] * 4,
        option_duration=1,
        max_highlevel_steps=5,
        scenario_set="continuous_pursuit",
        episode_lowlevel_steps=5,
        regime_duration=2,
    )
    recorder = TrajectoryRecorder(env)
    recorder.attach()
    result = rollout_episode(
        env,
        recorder,
        episode_id=9,
        mode="fixed",
        fixed_policy=2,
        high_model=None,
        scenario_set="continuous_pursuit",
    )
    assert result.scenario == "continuous_pursuit"
    assert result.lowlevel_steps > 0
    assert result.regime_starts[0][1] == "rear"
    assert len(result.evader_points) >= result.lowlevel_steps + 1
