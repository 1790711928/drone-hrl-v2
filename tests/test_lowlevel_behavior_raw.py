import pytest

from src.evaluation.eval_lowlevel_behavior_raw import (
    compute_raw_behavior_metrics,
    late_window,
    early_window,
    non_closing_ratio,
    sample_paired_initial_state,
    summarize_rows,
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
