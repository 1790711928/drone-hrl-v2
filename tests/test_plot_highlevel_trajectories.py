import numpy as np

from src.evaluation.plot_highlevel_trajectories import (
    EpisodePlotData,
    TrajectoryRecorder,
    compressed_option_sequence,
    phase_reset_jumps,
    rollout_episode,
    auto_plot_bounds,
    boundary_priority_starts,
    sample_points,
    save_plot,
    select_episodes_for_plot,
    select_best_showcase_episodes,
    showcase_selection_metrics,
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
        boundary_priority_points=[],
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


def test_plot_sampling_and_boundary_priority_starts():
    points = [(float(i), 0.0, 0.0) for i in range(11)]
    assert sample_points(points, 5) == [points[0], points[5], points[10]]
    assert boundary_priority_starts([3, 4, 5, 10, 11]) == [3, 10]


def test_auto_plot_bounds_focuses_on_trajectory_extent():
    bounds = auto_plot_bounds([(0.0, 0.0, 0.0), (10.0, 1.0, 2.0)], [(2.0, 8.0, 1.0)])
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    assert x_min < 0.0 < x_max
    assert y_min < 0.0 < y_max
    assert z_min < 0.0 < z_max
    assert (x_max - x_min) == (y_max - y_min)


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


def test_selection_filters_for_showcase_quality():
    weak = episode(episode_id=1)
    weak.switch_count = 1
    weak.lowlevel_steps = 300
    weak.option_sequence = [0, 2]
    strong = episode(episode_id=2)
    strong.switch_count = 3
    strong.lowlevel_steps = 320
    strong.option_sequence = [0, 2, 1]
    selected = select_episodes_for_plot(
        [weak, strong],
        max_plots=2,
        fixed_policy=2,
        only_success=False,
        only_failure=False,
        min_switch_count=3,
        min_lowlevel_steps=250,
        min_unique_options=3,
    )
    assert [item.episode_id for item in selected] == [2]
    assert strong.summary_row()["unique_option_count"] == 3


def test_showcase_score_prefers_balanced_close_multi_strategy_rollout():
    weak = EpisodePlotData(
        scenario="scripted_showcase",
        mode="scripted_showcase",
        episode_id=1,
        outcome="out_of_bounds",
        success=False,
        switch_count=1,
        option_sequence=[0, 2],
        completed_phases=0,
        evader_points=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)],
        pursuer_points=[(-80.0, 0.0, 0.0), (-70.0, 0.0, 0.0), (-60.0, 0.0, 0.0)],
        option_switches=[(0, "pi1"), (1, "pi3")],
        phase_starts=[(0, "start")],
        regime_starts=[(0, "rear"), (1, "boundary")],
        boundary_priority_points=[],
        lowlevel_steps=500,
        showcase_seed=1,
    )
    strong = EpisodePlotData(
        scenario="scripted_showcase",
        mode="scripted_showcase",
        episode_id=2,
        outcome="escaped",
        success=True,
        switch_count=5,
        option_sequence=[0, 1, 3, 2],
        completed_phases=0,
        evader_points=[(0.0, 0.0, 0.0), (20.0, 5.0, 6.0), (35.0, 28.0, 12.0), (45.0, 40.0, 30.0), (60.0, 48.0, 36.0)],
        pursuer_points=[(-5.0, -2.0, -1.0), (16.0, 4.0, 5.0), (32.0, 24.0, 10.0), (42.0, 36.0, 28.0), (55.0, 44.0, 34.0)],
        option_switches=[(0, "pi1"), (1, "pi2"), (2, "pi4"), (3, "pi3")],
        phase_starts=[(0, "start")],
        regime_starts=[(0, "rear"), (1, "flank"), (2, "vertical"), (3, "boundary")],
        boundary_priority_points=[],
        lowlevel_steps=380,
        showcase_seed=2,
    )
    selected = select_best_showcase_episodes([weak, strong], max_plots=1)
    assert selected == [strong]
    assert strong.selected_flag
    assert strong.selected_rank == 1
    assert showcase_selection_metrics(strong)["selection_score"] > showcase_selection_metrics(weak)["selection_score"]
    assert strong.summary_row()["selected_flag"] == 1


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
    assert result.regime_starts[0][1] in {"rear", "boundary"}
    assert len(result.evader_points) >= result.lowlevel_steps + 1


def test_save_plot_supports_topdown_without_dense_text(tmp_path):
    ep = EpisodePlotData(
        scenario="continuous_showcase",
        mode="regime_oracle",
        episode_id=12,
        outcome="escaped",
        success=True,
        switch_count=2,
        option_sequence=[0, 2, 1],
        completed_phases=0,
        evader_points=[(0.0, 0.0, 10.0), (1.0, 1.0, 10.2), (2.0, 1.5, 10.4)],
        pursuer_points=[(-1.0, -1.0, 9.5), (0.0, 0.5, 9.7), (1.0, 1.0, 9.9)],
        option_switches=[(0, "pi1"), (1, "pi3"), (2, "pi2")],
        phase_starts=[(0, "start")],
        regime_starts=[(0, "rear"), (2, "flank")],
        boundary_priority_points=[1, 2],
        lowlevel_steps=3,
    )
    output = save_plot(
        ep,
        tmp_path,
        (-10, 10, -10, 10, 0, 20),
        plot_sample_rate=2,
        max_annotations=0,
        no_text_annotations=True,
        view="topdown",
    )
    assert output.exists()
    assert output.name.endswith("_topdown.png")


def test_scripted_showcase_rollout_uses_regime_to_strategy_mapping():
    env = HighLevelOptionEnv(
        [ZeroModel()] * 4,
        option_duration=1,
        max_highlevel_steps=20,
        scenario_set="scripted_showcase",
        episode_lowlevel_steps=170,
    )
    recorder = TrajectoryRecorder(env)
    recorder.attach()
    result = rollout_episode(
        env,
        recorder,
        episode_id=13,
        mode="scripted_showcase",
        fixed_policy=0,
        high_model=None,
        scenario_set="scripted_showcase",
    )
    assert result.scenario == "scripted_showcase"
    assert result.lowlevel_steps > 0
    assert len(set(result.option_sequence)) >= 3
    assert result.option_sequence[:3] == [0, 1, 3]


def test_scripted_showcase_summary_keeps_sequence_and_save_plot_callouts(tmp_path):
    ep = EpisodePlotData(
        scenario="scripted_showcase",
        mode="scripted_showcase",
        episode_id=14,
        outcome="escaped",
        success=True,
        switch_count=3,
        option_sequence=[0, 1, 3, 2],
        completed_phases=0,
        evader_points=[(0.0, 0.0, 150.0), (5.0, 5.0, 151.0), (10.0, 6.0, 155.0), (12.0, 10.0, 154.0)],
        pursuer_points=[(-2.0, -1.0, 149.0), (3.0, 3.0, 150.0), (8.0, 5.0, 153.0), (11.0, 8.0, 153.0)],
        option_switches=[(0, "pi1"), (1, "pi2"), (2, "pi4"), (3, "pi3")],
        phase_starts=[(0, "start")],
        regime_starts=[(0, "rear"), (1, "flank"), (2, "vertical"), (3, "boundary")],
        boundary_priority_points=[],
        lowlevel_steps=500,
    )
    row = ep.summary_row()
    assert row["option_sequence"] == "pi1->pi2->pi4->pi3"
    assert row["showcase_script"]
    output = save_plot(ep, tmp_path, (-20, 20, -20, 20, 0, 300), show_callouts=True, max_annotations=4)
    assert output.exists()
