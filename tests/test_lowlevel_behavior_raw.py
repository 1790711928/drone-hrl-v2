from pathlib import Path
import zipfile

import pytest

from src.evaluation.eval_lowlevel_behavior_raw import (
    absolute_trace_reduction,
    compute_raw_behavior_metrics,
    compute_task_valid,
    evader_path_length,
    fixed_reference_gaps,
    gated_metric,
    non_decreasing_ratio,
    rear_away_displacement_trace,
    rear_escape_efficiency,
    late_window,
    early_window,
    non_closing_ratio,
    sample_paired_initial_state,
    summarize_rows,
    trace_change,
    vertical_away_displacement_trace,
    write_behavior_workbook,
)
from src.env.dynamics import Agent3DState, Env3DState


def make_state(*, evader=(0.0, 0.0, 0.0), pursuer=(-5.0, 0.0, 0.0), yaw=0.0, pitch=0.0) -> Env3DState:
    return Env3DState(
        evader=Agent3DState(*evader, speed=10.0, yaw=yaw, pitch=pitch),
        pursuer=Agent3DState(*pursuer, speed=10.0, yaw=0.0, pitch=0.0),
    )


def test_non_closing_ratio_uses_environment_sign_convention() -> None:
    # Environment closing_speed = (previous_distance - current_distance) / dt.
    # Positive is closing; zero/negative is non-closing.
    assert non_closing_ratio([1.0, 0.0, -0.5]) == 2 / 3


def test_raw_metrics_are_unweighted_and_use_physical_vertical_separation() -> None:
    metrics = compute_raw_behavior_metrics(
        distances=[10.0, 11.0, 13.0],
        closing_speeds=[1.0, 0.0, -1.0],
        threat_right=[0.8, 0.6, 0.2, 0.1, 0.1],
        min_boundary_margins=[0.2, 0.3, 0.4],
        boundary_margin_z=[0.1, 0.2, 0.3],
        vertical_separations=[2.0, 3.0, 5.0],
        boundary_safe_threshold=0.35,
        vertical_safe_threshold=0.20,
    )
    assert metrics["distance_gain"] == 3.0
    assert metrics["non_closing_ratio"] == 2 / 3
    assert metrics["lateral_threat_reduction"] == pytest.approx(0.8 - 0.1)
    assert metrics["boundary_margin_gain"] == 0.2
    assert metrics["vertical_separation_gain"] == 3.0
    assert metrics["safe_boundary_ratio"] == 1 / 3
    assert metrics["vertical_safe_ratio"] == 2 / 3


def test_early_and_late_windows_are_explicit_and_bounded() -> None:
    values = [float(index) for index in range(10)]
    assert early_window(values, 0.2) == [0.0, 1.0]
    assert late_window(values, 0.2) == [8.0, 9.0]
    assert early_window([1.0], 0.2) == [1.0]
    assert late_window([1.0], 0.2) == [1.0]


def test_diagnostic_trace_change_is_signed_late_minus_early() -> None:
    result = trace_change([-1.0, -0.5, 0.5, 1.0, 1.5])
    assert result["early_mean"] == pytest.approx(-1.0)
    assert result["late_mean"] == pytest.approx(1.5)
    assert result["improvement"] == pytest.approx(2.5)


def test_rear_gap_is_positive_behind_and_increases_with_fixed_initial_axis() -> None:
    initial = make_state()
    farther_behind = make_state(pursuer=(-7.0, 0.0, 0.0))
    rear_gaps, _ = fixed_reference_gaps(initial, [initial, farther_behind])
    assert rear_gaps == pytest.approx([5.0, 7.0])
    assert non_decreasing_ratio(rear_gaps) == 1.0


def test_rear_gap_ignores_later_evader_heading_changes() -> None:
    initial = make_state()
    changed_heading = make_state(yaw=1.1, pitch=0.4)
    rear_gaps, _ = fixed_reference_gaps(initial, [initial, changed_heading])
    assert rear_gaps == pytest.approx([5.0, 5.0])


def test_side_gap_sigma_makes_left_and_right_threats_comparable() -> None:
    right_initial = make_state(pursuer=(0.0, 5.0, 0.0))
    left_initial = make_state(pursuer=(0.0, -5.0, 0.0))
    _, right_gap = fixed_reference_gaps(right_initial, [right_initial])
    _, left_gap = fixed_reference_gaps(left_initial, [left_initial])
    assert right_gap == pytest.approx([5.0])
    assert left_gap == pytest.approx([5.0])


def test_side_gap_increases_away_from_initial_side_and_ignores_heading() -> None:
    initial = make_state(pursuer=(0.0, 5.0, 0.0))
    moved_away = make_state(evader=(0.0, -2.0, 0.0), pursuer=(0.0, 5.0, 0.0), yaw=1.2, pitch=-0.3)
    _, side_gaps = fixed_reference_gaps(initial, [initial, moved_away])
    assert side_gaps == pytest.approx([5.0, 7.0])


def test_rear_away_displacement_prefers_threat_away_motion_over_side_motion() -> None:
    initial = make_state()
    away = make_state(evader=(2.0, 0.0, 0.0))
    side = make_state(evader=(0.0, 2.0, 0.0))
    assert rear_away_displacement_trace(initial, [initial, away])[-1] == pytest.approx(2.0)
    assert rear_away_displacement_trace(initial, [initial, side])[-1] == pytest.approx(0.0)


def test_rear_escape_efficiency_is_direct_one_and_reverse_negative() -> None:
    initial = make_state()
    direct = [initial, make_state(evader=(1.0, 0.0, 0.0)), make_state(evader=(2.0, 0.0, 0.0))]
    winding = [
        initial,
        make_state(evader=(0.5, 1.0, 0.0)),
        make_state(evader=(1.0, 0.0, 0.0)),
        make_state(evader=(2.0, 0.0, 0.0)),
    ]
    reverse = [initial, make_state(evader=(-2.0, 0.0, 0.0))]
    assert rear_escape_efficiency(initial, direct)[2] == pytest.approx(1.0)
    assert rear_escape_efficiency(initial, direct)[2] > rear_escape_efficiency(initial, winding)[2]
    assert rear_escape_efficiency(initial, reverse)[2] < 0.0
    assert evader_path_length(direct) == pytest.approx(2.0)


def test_vertical_away_displacement_is_positive_away_from_upper_or_lower_threat() -> None:
    upper_threat = make_state(evader=(0.0, 0.0, 0.0), pursuer=(-5.0, 0.0, 5.0))
    lower_threat = make_state(evader=(0.0, 0.0, 0.0), pursuer=(-5.0, 0.0, -5.0))
    down = make_state(evader=(0.0, 0.0, -2.0), pursuer=(-5.0, 0.0, 5.0))
    up = make_state(evader=(0.0, 0.0, 2.0), pursuer=(-5.0, 0.0, -5.0))
    assert vertical_away_displacement_trace(upper_threat, [upper_threat, down])[-1] == pytest.approx(2.0)
    assert vertical_away_displacement_trace(lower_threat, [lower_threat, up])[-1] == pytest.approx(2.0)


def test_vertical_away_displacement_is_mirror_symmetric() -> None:
    upper = make_state(evader=(0.0, 0.0, 0.0), pursuer=(-5.0, 0.0, 5.0))
    upper_down = make_state(evader=(0.0, 0.0, -2.0), pursuer=(-5.0, 0.0, 5.0))
    lower = make_state(evader=(0.0, 0.0, 0.0), pursuer=(-5.0, 0.0, -5.0))
    lower_up = make_state(evader=(0.0, 0.0, 2.0), pursuer=(-5.0, 0.0, -5.0))
    assert vertical_away_displacement_trace(upper, [upper, upper_down])[-1] == pytest.approx(
        vertical_away_displacement_trace(lower, [lower, lower_up])[-1]
    )


def test_task_valid_and_gated_metric_keep_failures_as_zero_contribution() -> None:
    assert compute_task_valid("escaped") == 1
    for outcome in ("captured", "out_of_bounds", "timeout", "other"):
        assert compute_task_valid(outcome) == 0
    assert gated_metric(0, 3.5) == 0.0
    assert gated_metric(1, 3.5) == 3.5


def test_vertical_threat_reduction_is_mirror_symmetric() -> None:
    positive = absolute_trace_reduction([0.8, 0.4, 0.2])
    mirrored = absolute_trace_reduction([-0.8, -0.4, -0.2])
    assert positive["reduction"] == pytest.approx(mirrored["reduction"])


def test_paired_initial_state_is_reproducible() -> None:
    first = sample_paired_initial_state("rear_close_threat", 17)
    second = sample_paired_initial_state("rear_close_threat", 17)
    assert first == second


def test_summary_reports_mean_std_median_min_max_and_outcome_rates() -> None:
    base = {
        "scenario": "rear_close_threat",
        "policy": "pi1",
        "success": 1,
        "captured": 0,
        "out_of_bounds": 0,
        "timeout": 0,
    }
    rows = [
        {**base, "distance_gain": 1.0, "non_closing_ratio": 0.5, "lateral_threat_reduction": 0.0, "boundary_margin_gain": 0.0, "safe_boundary_ratio": 1.0, "vertical_separation_gain": 1.0, "vertical_safe_ratio": 1.0},
        {**base, "distance_gain": 3.0, "non_closing_ratio": 1.0, "lateral_threat_reduction": 0.0, "boundary_margin_gain": 0.0, "safe_boundary_ratio": 1.0, "vertical_separation_gain": 3.0, "vertical_safe_ratio": 1.0},
    ]
    summary = summarize_rows(rows)
    distance = next(item for item in summary if item["metric"] == "distance_gain")
    assert distance["mean"] == 2.0
    assert distance["median"] == 2.0
    assert distance["min"] == 1.0
    assert distance["max"] == 3.0
    assert distance["episode_count"] == 2
    assert distance["success_rate"] == 1.0


def test_summary_can_include_new_diagnostic_metrics() -> None:
    row = {
        "scenario": "rear_close_threat",
        "policy": "pi1",
        "success": 1,
        "captured": 0,
        "out_of_bounds": 0,
        "timeout": 0,
        "rear_gap_gain": 2.5,
    }
    summary = summarize_rows([row], metrics=["rear_gap_gain"])
    assert summary[0]["metric"] == "rear_gap_gain"
    assert summary[0]["mean"] == 2.5


def test_summary_keeps_raw_mean_and_valid_mean_together() -> None:
    row = {
        "scenario": "rear_close_threat",
        "policy": "pi1",
        "success": 0,
        "captured": 0,
        "out_of_bounds": 0,
        "timeout": 1,
        "distance_gain": 4.0,
        "distance_gain_valid": 0.0,
    }
    summary = summarize_rows([row], metrics=["distance_gain"])
    assert summary[0]["mean"] == 4.0
    assert summary[0]["valid_mean"] == 0.0
    assert summary[0]["success_rate"] == 0.0


def test_workbook_contains_required_sheets_and_manifest(tmp_path: Path) -> None:
    path = tmp_path / "behavior_evaluation.xlsx"
    write_behavior_workbook(
        path,
        raw_rows=[],
        summary_rows=[],
        model_paths=[Path("pi1.zip"), Path("pi2.zip"), Path("pi3.zip"), Path("pi4.zip")],
        boundary_safe_threshold=0.35,
        vertical_safe_threshold=0.20,
    )
    with zipfile.ZipFile(path) as archive:
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
        assert 'name="Summary"' in workbook
        assert 'name="Manifest"' in workbook
        assert 'name="Rear"' in workbook
        assert 'name="Flank"' in workbook
        assert 'name="Boundary"' in workbook
        assert 'name="Vertical"' in workbook
        assert archive.testzip() is None
