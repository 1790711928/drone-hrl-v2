import csv
from pathlib import Path

import pytest

from src.env.dynamics import Agent3DState, Env3DState
from src.evaluation.export_rear_trajectory_data import (
    INDEX_FIELDS,
    POLICIES,
    SCENARIO,
    TRAJECTORY_FIELDS,
    _backfill_outcome,
    _make_row,
    initial_state_values,
    observation_dict_from_flat,
    physical_action,
    state_hash,
    validate_csv_round_trip,
    validate_export,
    write_csv,
)
from src.training.sac_env import PursuitEscapeGymEnv


def make_state() -> Env3DState:
    return Env3DState(
        evader=Agent3DState(0.0, 0.0, 10.0, 10.0, 0.2, 0.0),
        pursuer=Agent3DState(-6.0, 0.0, 10.0, 11.5, 0.0, 0.0),
    )


def make_obs() -> dict[str, float]:
    return {key: 0.0 for key in PursuitEscapeGymEnv.OBS_KEYS}


def test_observation_export_has_all_named_25d_components() -> None:
    flat = [float(index) for index in range(25)]
    result = observation_dict_from_flat(flat)
    assert len(result) == 25
    assert result["threat_forward"] == 22.0
    assert result["threat_right"] == 23.0
    assert result["threat_up"] == 24.0


def test_physical_action_matches_current_gym_wrapper_conversion() -> None:
    env = PursuitEscapeGymEnv(scenario=SCENARIO, randomize_reset=False)
    assert physical_action([1.0, -0.5, 0.5], env) == pytest.approx((1.0, -0.5, 0.35))


def test_initial_row_and_outcome_backfill() -> None:
    state = make_state()
    initial_hash = state_hash(state)
    rows = [
        _make_row(
            state=state,
            obs_dict=make_obs(),
            initial_state=state,
            initial_hash=initial_hash,
            policy="pi1",
            seed=20261008,
            trial_id=0,
            step=0,
            action=("", "", ""),
            step_reward=0.0,
            cumulative_reward=0.0,
        )
    ]
    _backfill_outcome(rows, "escaped", 0)
    assert rows[0]["is_initial_state"] == 1
    assert rows[0]["action_accel"] == ""
    assert rows[0]["outcome"] == "escaped"
    assert rows[0]["success"] == 1
    assert rows[0]["normalized_time"] == 0.0


def test_paired_validation_checks_steps_hash_and_csv_round_trip(tmp_path: Path) -> None:
    state = make_state()
    initial_hash = state_hash(state)
    all_rows = []
    index_rows = []
    for policy in POLICIES:
        rows = [
            _make_row(
                state=state,
                obs_dict=make_obs(),
                initial_state=state,
                initial_hash=initial_hash,
                policy=policy,
                seed=20261008,
                trial_id=0,
                step=0,
                action=("", "", ""),
                step_reward=0.0,
                cumulative_reward=0.0,
            ),
            _make_row(
                state=state,
                obs_dict=make_obs(),
                initial_state=state,
                initial_hash=initial_hash,
                policy=policy,
                seed=20261008,
                trial_id=0,
                step=1,
                action=(0.0, 0.0, 0.0),
                step_reward=1.0,
                cumulative_reward=1.0,
            ),
        ]
        _backfill_outcome(rows, "escaped", 1)
        all_rows.extend(rows)
        index_rows.append(
            {
                "scenario": SCENARIO,
                "policy": policy,
                "seed": 20261008,
                "trial_id": 0,
                "outcome": "escaped",
                "success": 1,
                "episode_length": 1,
                "initial_distance": rows[0]["distance"],
                "final_distance": rows[-1]["distance"],
                "total_reward": 1.0,
                "checkpoint_path": "/tmp/checkpoint.zip",
                "initial_state_hash": initial_hash,
            }
        )
    validate_export(all_rows, index_rows, episodes=1)
    trajectory_path = tmp_path / "rear_trajectory_steps.csv"
    index_path = tmp_path / "rear_episode_index.csv"
    write_csv(trajectory_path, TRAJECTORY_FIELDS, all_rows)
    write_csv(index_path, INDEX_FIELDS, index_rows)
    validate_csv_round_trip(trajectory_path, TRAJECTORY_FIELDS, len(all_rows))
    validate_csv_round_trip(index_path, INDEX_FIELDS, len(index_rows))
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 8
    assert len(initial_state_values(state)) == 12
