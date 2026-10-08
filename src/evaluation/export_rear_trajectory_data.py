"""Export paired, step-level raw trajectories for the rear scenario.

This tool intentionally performs no behavior scoring or feature selection.  It
only records the environment state, observation, action, and reward for the
four low-level policies from one shared initial state per trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.env.dynamics import Env3DState, relative_distance
from src.evaluation.eval_lowlevel_behavior_raw import (
    _paired_env,
    _state_observation,
    sample_paired_initial_state,
)
from src.training.sac_env import PursuitEscapeGymEnv

SCENARIO = "rear_close_threat"
POLICIES = ["pi1", "pi2", "pi3", "pi4"]
DEFAULT_EPISODES = 5
DEFAULT_SEED = 20261008
DEFAULT_OUTPUT_DIR = "outputs/paper_eval_core/rear_trajectory_export"
DEFAULT_MODEL_PATHS = [
    "outputs/checkpoints/specialist_fix/sac_low_1_rear_close_threat_specialist_fix.zip",
    "outputs/checkpoints/sac_low_2_flank_threat.zip",
    "outputs/checkpoints/sac_low_3_boundary_constrained.zip",
    "outputs/checkpoints/specialist_fix/sac_low_4_vertical_z_threat_specialist_fix.zip",
]
MODEL_RELATIVE_PATHS = [
    "specialist_fix/sac_low_1_rear_close_threat_specialist_fix.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "specialist_fix/sac_low_4_vertical_z_threat_specialist_fix.zip",
]
OBS_EXPORT_FIELDS = [f"obs_{key}" for key in PursuitEscapeGymEnv.OBS_KEYS]
INITIAL_FIELDS = [
    "initial_evader_x",
    "initial_evader_y",
    "initial_evader_z",
    "initial_evader_speed",
    "initial_evader_yaw",
    "initial_evader_pitch",
    "initial_pursuer_x",
    "initial_pursuer_y",
    "initial_pursuer_z",
    "initial_pursuer_speed",
    "initial_pursuer_yaw",
    "initial_pursuer_pitch",
]
TRAJECTORY_FIELDS = [
    "scenario",
    "policy",
    "seed",
    "trial_id",
    "step",
    "normalized_time",
    "is_initial_state",
    "outcome",
    "success",
    "captured",
    "out_of_bounds",
    "timeout",
    "episode_length",
    "evader_x",
    "evader_y",
    "evader_z",
    "evader_speed",
    "evader_yaw",
    "evader_pitch",
    "pursuer_x",
    "pursuer_y",
    "pursuer_z",
    "pursuer_speed",
    "pursuer_yaw",
    "pursuer_pitch",
    "relative_x",
    "relative_y",
    "relative_z",
    "distance",
    "horizontal_distance",
    "vertical_separation",
    "closing_speed",
    "threat_forward",
    "threat_right",
    "threat_up",
    "boundary_margin_x",
    "boundary_margin_y",
    "boundary_margin_z",
    "min_boundary_margin",
    "normalized_step",
    "action_accel",
    "action_yaw_rate",
    "action_pitch_rate",
    "step_reward",
    "cumulative_reward",
    "initial_state_hash",
    *INITIAL_FIELDS,
    *OBS_EXPORT_FIELDS,
]
INDEX_FIELDS = [
    "scenario",
    "policy",
    "seed",
    "trial_id",
    "outcome",
    "success",
    "episode_length",
    "initial_distance",
    "final_distance",
    "total_reward",
    "checkpoint_path",
    "initial_state_hash",
]


def state_hash(state: Env3DState) -> str:
    """Create a stable hash from the complete paired initial state."""
    values = [
        state.evader.x,
        state.evader.y,
        state.evader.z,
        state.evader.speed,
        state.evader.yaw,
        state.evader.pitch,
        state.pursuer.x,
        state.pursuer.y,
        state.pursuer.z,
        state.pursuer.speed,
        state.pursuer.yaw,
        state.pursuer.pitch,
    ]
    payload = "|".join(f"{float(value):.17g}" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initial_state_values(state: Env3DState) -> dict[str, float]:
    return {
        "initial_evader_x": state.evader.x,
        "initial_evader_y": state.evader.y,
        "initial_evader_z": state.evader.z,
        "initial_evader_speed": state.evader.speed,
        "initial_evader_yaw": state.evader.yaw,
        "initial_evader_pitch": state.evader.pitch,
        "initial_pursuer_x": state.pursuer.x,
        "initial_pursuer_y": state.pursuer.y,
        "initial_pursuer_z": state.pursuer.z,
        "initial_pursuer_speed": state.pursuer.speed,
        "initial_pursuer_yaw": state.pursuer.yaw,
        "initial_pursuer_pitch": state.pursuer.pitch,
    }


def observation_dict_from_flat(obs: Sequence[float]) -> dict[str, float]:
    if len(obs) != len(PursuitEscapeGymEnv.OBS_KEYS):
        raise ValueError(f"expected {len(PursuitEscapeGymEnv.OBS_KEYS)} observation values, got {len(obs)}")
    return {key: float(obs[index]) for index, key in enumerate(PursuitEscapeGymEnv.OBS_KEYS)}


def physical_action(action: Sequence[float], env: PursuitEscapeGymEnv) -> tuple[float, float, float]:
    """Convert policy output to the physical action passed to inner env.step."""
    if len(action) < 3:
        raise ValueError("policy action must contain accel, yaw_rate, and pitch_rate")
    return (
        float(action[0]) * 1.0,
        float(action[1]) * env.inner.env_cfg.yaw_rate_max,
        float(action[2]) * env.inner.env_cfg.pitch_rate_max,
    )


def _state_row_values(state: Env3DState, obs_dict: dict[str, float]) -> dict[str, float]:
    relative_x = state.pursuer.x - state.evader.x
    relative_y = state.pursuer.y - state.evader.y
    relative_z = state.pursuer.z - state.evader.z
    return {
        "evader_x": state.evader.x,
        "evader_y": state.evader.y,
        "evader_z": state.evader.z,
        "evader_speed": state.evader.speed,
        "evader_yaw": state.evader.yaw,
        "evader_pitch": state.evader.pitch,
        "pursuer_x": state.pursuer.x,
        "pursuer_y": state.pursuer.y,
        "pursuer_z": state.pursuer.z,
        "pursuer_speed": state.pursuer.speed,
        "pursuer_yaw": state.pursuer.yaw,
        "pursuer_pitch": state.pursuer.pitch,
        "relative_x": relative_x,
        "relative_y": relative_y,
        "relative_z": relative_z,
        "distance": relative_distance(state),
        "horizontal_distance": math.hypot(relative_x, relative_y),
        "vertical_separation": abs(relative_z),
        "closing_speed": obs_dict["closing_speed"],
        "threat_forward": obs_dict["threat_forward"],
        "threat_right": obs_dict["threat_right"],
        "threat_up": obs_dict["threat_up"],
        "boundary_margin_x": obs_dict["boundary_margin_x"],
        "boundary_margin_y": obs_dict["boundary_margin_y"],
        "boundary_margin_z": obs_dict["boundary_margin_z"],
        "min_boundary_margin": obs_dict["min_boundary_margin"],
        "normalized_step": obs_dict["normalized_step"],
    }


def _make_row(
    *,
    state: Env3DState,
    obs_dict: dict[str, float],
    initial_state: Env3DState,
    initial_hash: str,
    policy: str,
    seed: int,
    trial_id: int,
    step: int,
    action: tuple[float | str, float | str, float | str],
    step_reward: float,
    cumulative_reward: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": SCENARIO,
        "policy": policy,
        "seed": seed,
        "trial_id": trial_id,
        "step": step,
        "normalized_time": "",
        "is_initial_state": int(step == 0),
        "outcome": "",
        "success": "",
        "captured": "",
        "out_of_bounds": "",
        "timeout": "",
        "episode_length": "",
        **_state_row_values(state, obs_dict),
        "action_accel": action[0],
        "action_yaw_rate": action[1],
        "action_pitch_rate": action[2],
        "step_reward": step_reward,
        "cumulative_reward": cumulative_reward,
        "initial_state_hash": initial_hash,
        **initial_state_values(initial_state),
    }
    row.update({f"obs_{key}": value for key, value in obs_dict.items()})
    return row


def _backfill_outcome(rows: list[dict[str, Any]], outcome: str, episode_length: int) -> None:
    success = int(outcome == "escaped")
    for row in rows:
        row.update(
            {
                "normalized_time": row["step"] / episode_length if episode_length else 0.0,
                "outcome": outcome,
                "success": success,
                "captured": int(outcome == "captured"),
                "out_of_bounds": int(outcome == "out_of_bounds"),
                "timeout": int(outcome == "timeout"),
                "episode_length": episode_length,
            }
        )


def run_episode(model: Any, *, policy: str, seed: int, trial_id: int, initial_state: Env3DState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one policy from the supplied paired state and return rows plus index."""
    env = _paired_env(SCENARIO, initial_state)
    state = env.inner.state
    assert state is not None
    initial_hash = state_hash(initial_state)
    obs_dict = _state_observation(env, 0.0)
    obs = env._flatten_obs(obs_dict)
    cumulative_reward = 0.0
    rows = [
        _make_row(
            state=state,
            obs_dict=obs_dict,
            initial_state=initial_state,
            initial_hash=initial_hash,
            policy=policy,
            seed=seed,
            trial_id=trial_id,
            step=0,
            action=("", "", ""),
            step_reward=0.0,
            cumulative_reward=0.0,
        )
    ]
    terminated = truncated = False
    info: dict[str, Any] = {"outcome": "timeout"}
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_values = [float(value) for value in action]
        physical = physical_action(action_values, env)
        obs, reward, terminated, truncated, info = env.step(action_values)
        step += 1
        cumulative_reward += float(reward)
        state = env.inner.state
        assert state is not None
        obs_dict = observation_dict_from_flat(obs)
        rows.append(
            _make_row(
                state=state,
                obs_dict=obs_dict,
                initial_state=initial_state,
                initial_hash=initial_hash,
                policy=policy,
                seed=seed,
                trial_id=trial_id,
                step=step,
                action=physical,
                step_reward=float(reward),
                cumulative_reward=cumulative_reward,
            )
        )

    outcome = str(info.get("outcome", "timeout"))
    _backfill_outcome(rows, outcome, step)
    final_state = env.inner.state
    assert final_state is not None
    index_row = {
        "scenario": SCENARIO,
        "policy": policy,
        "seed": seed,
        "trial_id": trial_id,
        "outcome": outcome,
        "success": int(outcome == "escaped"),
        "episode_length": step,
        "initial_distance": rows[0]["distance"],
        "final_distance": rows[-1]["distance"],
        "total_reward": cumulative_reward,
        "checkpoint_path": "",
        "initial_state_hash": initial_hash,
    }
    return rows, index_row


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def validate_export(rows: Sequence[dict[str, Any]], index_rows: Sequence[dict[str, Any]], episodes: int) -> None:
    """Perform only structural checks required for a trajectory export."""
    if len(index_rows) != episodes * len(POLICIES):
        raise AssertionError(f"expected {episodes * len(POLICIES)} episodes, got {len(index_rows)}")
    hashes_by_trial: dict[int, set[str]] = {}
    policies_by_trial: dict[int, set[str]] = {}
    for row in index_rows:
        hashes_by_trial.setdefault(int(row["trial_id"]), set()).add(str(row["initial_state_hash"]))
        policies_by_trial.setdefault(int(row["trial_id"]), set()).add(str(row["policy"]))
    if any(len(hashes) != 1 for hashes in hashes_by_trial.values()):
        raise AssertionError("paired initial_state_hash mismatch within a trial")
    if any(policies != set(POLICIES) for policies in policies_by_trial.values()):
        raise AssertionError("each trial must contain pi1, pi2, pi3, and pi4")
    for row in index_rows:
        group = [item for item in rows if item["trial_id"] == row["trial_id"] and item["policy"] == row["policy"]]
        expected_steps = list(range(int(row["episode_length"]) + 1))
        if [int(item["step"]) for item in group] != expected_steps:
            raise AssertionError(f"non-contiguous steps for {row['policy']} trial {row['trial_id']}")
        if not group[0]["is_initial_state"] or int(group[0]["step"]) != 0:
            raise AssertionError("step zero is not marked as the initial state")
        if group[-1]["outcome"] != row["outcome"]:
            raise AssertionError("final trajectory outcome does not match episode index")
        for item in group:
            for field, value in item.items():
                if field in {"action_accel", "action_yaw_rate", "action_pitch_rate"} and item["step"] == 0:
                    continue
                if _is_nan(value):
                    raise AssertionError(f"NaN in trajectory field {field}")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_csv_round_trip(path: Path, fieldnames: Sequence[str], expected_rows: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fieldnames):
            raise AssertionError(f"unexpected CSV fields in {path}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows in {path}, got {len(rows)}")


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.policy_checkpoints is not None:
        return [Path(value) for value in args.policy_checkpoints]
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        return [checkpoint_dir / relative for relative in MODEL_RELATIVE_PATHS]
    return [Path(value) for value in DEFAULT_MODEL_PATHS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired raw rear trajectories")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--policy-checkpoints", nargs=4, default=None, metavar=("PI1", "PI2", "PI3", "PI4"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_paths = resolve_model_paths(args)
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing checkpoint: {path}")
        raise SystemExit("Provide the four requested checkpoints before running this exporter.")

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required to load SAC checkpoints.") from exc

    models = [SAC.load(str(path), device=args.device) for path in model_paths]
    trajectory_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for trial_id in range(args.episodes):
        trial_seed = args.seed + trial_id
        initial_state = sample_paired_initial_state(SCENARIO, trial_seed)
        for policy, model, checkpoint_path in zip(POLICIES, models, model_paths):
            rows, index_row = run_episode(
                model,
                policy=policy,
                seed=trial_seed,
                trial_id=trial_id,
                initial_state=initial_state,
            )
            index_row["checkpoint_path"] = str(checkpoint_path.resolve())
            trajectory_rows.extend(rows)
            index_rows.append(index_row)

    validate_export(trajectory_rows, index_rows, args.episodes)
    out_dir = Path(args.output_dir)
    trajectory_path = out_dir / "rear_trajectory_steps.csv"
    index_path = out_dir / "rear_episode_index.csv"
    write_csv(trajectory_path, TRAJECTORY_FIELDS, trajectory_rows)
    write_csv(index_path, INDEX_FIELDS, index_rows)
    validate_csv_round_trip(trajectory_path, TRAJECTORY_FIELDS, len(trajectory_rows))
    validate_csv_round_trip(index_path, INDEX_FIELDS, len(index_rows))
    print(f"[csv] trajectory: {trajectory_path}")
    print(f"[csv] episode index: {index_path}")
    print(f"episodes={len(index_rows)}, trajectory_rows={len(trajectory_rows)}")


if __name__ == "__main__":
    main()
