"""Independent paired evaluator for the reconstructed Chapter 3 metrics.

This file intentionally does not import any existing score/alignment evaluator.
It reuses only the environment/model interfaces and writes only below its own
``reconstructed_metric_system`` directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml


POLICIES = ["pi1", "pi2", "pi3", "pi4"]
RAW_FIELDS = [
    "M_turn",
    "G_v",
    "G_s",
    "R_s",
    "G_b",
    "R_b",
    "G_z",
    "G_d",
]
CSV_FIELDS = [
    "scenario",
    "scenario_label",
    "target_policy",
    "policy",
    "episode_index",
    "seed",
    "outcome",
    "task_valid_escaped_only",
    "episode_steps",
    *RAW_FIELDS,
    "initial_distance",
    "final_distance",
    "initial_evader_speed",
    "final_evader_speed",
    "initial_side_gap_norm",
    "final_side_gap_norm",
    "initial_boundary_margin_norm",
    "final_boundary_margin_norm",
    "initial_vertical_separation",
    "final_vertical_separation",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def refuse_nonempty_output(path: Path) -> None:
    resolved = path.resolve()
    if BASE_DIR.resolve() not in resolved.parents:
        raise ValueError(f"output must remain under {BASE_DIR}, got {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty output directory: {resolved}; choose a new run directory"
        )
    resolved.mkdir(parents=True, exist_ok=True)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def window(values: list[float], fraction: float, *, late: bool) -> list[float]:
    if not values:
        return []
    count = max(1, int(len(values) * fraction))
    return values[-count:] if late else values[:count]


def non_decreasing_ratio(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(cur >= prev for prev, cur in zip(values, values[1:])) / (len(values) - 1)


def initial_state_for(scenario: str, seed: int) -> Env3DState:
    """Use the existing environment reset to create one reproducible state."""
    from src.env.pursuit_escape_env import PursuitEscapeEnv

    reference = PursuitEscapeEnv()
    reference.reset(scenario=scenario, randomize=True, rng=random.Random(seed))
    if reference.state is None:
        raise RuntimeError("environment reset did not produce an initial state")
    return copy.deepcopy(reference.state)


def paired_env(scenario: str, initial_state: Env3DState) -> PursuitEscapeGymEnv:
    """Inject a copied state into the existing Gym adapter without random reset."""
    from src.env.termination import TerminationState
    from src.training.sac_env import PursuitEscapeGymEnv

    env = PursuitEscapeGymEnv(scenario=scenario, randomize_reset=False)
    env.inner.state = copy.deepcopy(initial_state)
    env.inner.tstate = TerminationState()
    env.inner.current_scenario = scenario
    return env


def fixed_side_gap_norm(initial_state: Env3DState, state: Env3DState) -> float:
    """Signed lateral gap in the initial evader frame, scaled by initial range."""
    from src.env.dynamics import relative_distance

    ev0 = initial_state.evader
    pu0 = initial_state.pursuer
    right = (-math.sin(ev0.yaw), math.cos(ev0.yaw), 0.0)
    initial_rel = (pu0.x - ev0.x, pu0.y - ev0.y, pu0.z - ev0.z)
    initial_projection = sum(a * b for a, b in zip(initial_rel, right))
    sign = 1.0 if initial_projection >= 0.0 else -1.0
    rel = (state.pursuer.x - state.evader.x, state.pursuer.y - state.evader.y, state.pursuer.z - state.evader.z)
    projection = sign * sum(a * b for a, b in zip(rel, right))
    initial_distance = max(relative_distance(initial_state), 1.0)
    return projection / initial_distance


def episode_metrics(
    *,
    initial_state: Env3DState,
    state_trace: list[Env3DState],
    forward_trace: list[float],
    right_trace: list[float],
    speed_trace: list[float],
    side_gap_trace: list[float],
    distance_trace: list[float],
    boundary_trace: list[float],
    vertical_trace: list[float],
    safe_boundary_margin: float,
    early_fraction: float,
    late_fraction: float,
) -> dict[str, float]:
    """Compute the eight raw indicators before any validity or normalization."""
    early_speed = mean(window(speed_trace, early_fraction, late=False))
    late_speed = mean(window(speed_trace, late_fraction, late=True))
    early_side = mean(window(side_gap_trace, early_fraction, late=False))
    late_side = mean(window(side_gap_trace, late_fraction, late=True))
    return {
        # M_turn follows the paper's q_r/q_f concept using the environment's
        # normalized local threat projections. Full-trace aggregation is an
        # explicit reconstruction assumption.
        "M_turn": mean(abs(right) * max(-forward, 0.0) for forward, right in zip(forward_trace, right_trace)),
        # G_v is intentionally a late-vs-early speed change, not mean speed.
        "G_v": late_speed - early_speed,
        "G_s": late_side - early_side,
        "R_s": non_decreasing_ratio(side_gap_trace),
        "G_b": boundary_trace[-1] - boundary_trace[0],
        "R_b": mean(float(value >= safe_boundary_margin) for value in boundary_trace),
        "G_z": vertical_trace[-1] - vertical_trace[0],
        "G_d": distance_trace[-1] - distance_trace[0],
    }


def run_episode(
    *,
    model: Any,
    scenario: str,
    scenario_label: str,
    target_policy: str,
    policy: str,
    episode_index: int,
    seed: int,
    config: dict[str, Any],
    initial_state: Env3DState,
) -> dict[str, Any]:
    from src.env.dynamics import relative_distance

    evaluation = config["evaluation"]
    threshold = float(config["thresholds"]["safe_boundary_margin"])
    early_fraction = float(evaluation["early_fraction"])
    late_fraction = float(evaluation["late_fraction"])

    env = paired_env(scenario, initial_state)
    try:
        state = env.inner.state
        assert state is not None
        obs_dict = env.inner._observation(closing_speed=0.0)
        obs = env._flatten_obs(obs_dict)
        state_trace = [copy.deepcopy(state)]
        forward_trace = [float(obs_dict["threat_forward"])]
        right_trace = [float(obs_dict["threat_right"])]
        speed_trace = [float(state.evader.speed)]
        side_gap_trace = [fixed_side_gap_norm(initial_state, state)]
        distance_trace = [relative_distance(state)]
        boundary_trace = [float(obs_dict["min_boundary_margin"])]
        vertical_trace = [abs(state.evader.z - state.pursuer.z)]
        episode_steps = 0
        info: dict[str, Any] = {"outcome": "timeout"}
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            state = env.inner.state
            assert state is not None
            obs_dict = env.inner._observation(closing_speed=float(info.get("closing_speed", 0.0)))
            state_trace.append(copy.deepcopy(state))
            forward_trace.append(float(obs_dict["threat_forward"]))
            right_trace.append(float(obs_dict["threat_right"]))
            speed_trace.append(float(state.evader.speed))
            side_gap_trace.append(fixed_side_gap_norm(initial_state, state))
            distance_trace.append(relative_distance(state))
            boundary_trace.append(float(obs_dict["min_boundary_margin"]))
            vertical_trace.append(abs(state.evader.z - state.pursuer.z))
            episode_steps += 1
    finally:
        env.close()

    outcome = str(info.get("outcome", "timeout"))
    metrics = episode_metrics(
        initial_state=initial_state,
        state_trace=state_trace,
        forward_trace=forward_trace,
        right_trace=right_trace,
        speed_trace=speed_trace,
        side_gap_trace=side_gap_trace,
        distance_trace=distance_trace,
        boundary_trace=boundary_trace,
        vertical_trace=vertical_trace,
        safe_boundary_margin=threshold,
        early_fraction=early_fraction,
        late_fraction=late_fraction,
    )
    return {
        "scenario": scenario,
        "scenario_label": scenario_label,
        "target_policy": target_policy,
        "policy": policy,
        "episode_index": episode_index,
        "seed": seed,
        "outcome": outcome,
        "task_valid_escaped_only": int(outcome == "escaped"),
        "episode_steps": episode_steps,
        **metrics,
        "initial_distance": distance_trace[0],
        "final_distance": distance_trace[-1],
        "initial_evader_speed": speed_trace[0],
        "final_evader_speed": speed_trace[-1],
        "initial_side_gap_norm": side_gap_trace[0],
        "final_side_gap_norm": side_gap_trace[-1],
        "initial_boundary_margin_norm": boundary_trace[0],
        "final_boundary_margin_norm": boundary_trace[-1],
        "initial_vertical_separation": vertical_trace[0],
        "final_vertical_separation": vertical_trace[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "metric_config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    for index in range(1, 5):
        parser.add_argument(
            f"--policy{index}-checkpoint",
            dest=f"policy{index}_checkpoint",
            type=Path,
            required=True,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import the model library only for an actual evaluation. This keeps
    # ``--help`` and configuration inspection usable in an audit-only Python
    # environment that does not have Stable-Baselines3 installed.
    from stable_baselines3 import SAC

    config = load_yaml(args.config)
    episodes = int(args.episodes if args.episodes is not None else config["evaluation"]["episodes"])
    base_seed = int(args.seed if args.seed is not None else config["evaluation"]["seed"])
    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    refuse_nonempty_output(args.output_dir)

    scenario_specs = config["scenarios"]
    if [spec["target_policy"] for spec in scenario_specs] != POLICIES:
        raise ValueError("configuration must map scenarios to pi1, pi2, pi3, pi4 in order")
    model_paths = [getattr(args, f"policy{index}_checkpoint") for index in range(1, 5)]
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {path}")

    models = [SAC.load(str(path), device=args.device) for path in model_paths]
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario_spec in enumerate(scenario_specs):
        scenario = str(scenario_spec["id"])
        scenario_label = str(scenario_spec["label"])
        target_policy = str(scenario_spec["target_policy"])
        for episode_index in range(episodes):
            episode_seed = base_seed + scenario_index * 100000 + episode_index
            initial_state = initial_state_for(scenario, episode_seed)
            for policy, model in zip(POLICIES, models):
                rows.append(
                    run_episode(
                        model=model,
                        scenario=scenario,
                        scenario_label=scenario_label,
                        target_policy=target_policy,
                        policy=policy,
                        episode_index=episode_index,
                        seed=episode_seed,
                        config=config,
                        initial_state=initial_state,
                    )
                )

    raw_path = args.output_dir / "episodes_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema": "reconstructed_metric_system_v1",
        "status": "raw_evaluation_complete",
        "git_commit_at_run": git_commit(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "episodes": episodes,
        "base_seed": base_seed,
        "device": args.device,
        "paired_initial_state": True,
        "deterministic_action": True,
        "model_paths": [str(path.resolve()) for path in model_paths],
        "model_sha256": {policy: sha256_file(path) for policy, path in zip(POLICIES, model_paths)},
        "raw_csv": str(raw_path.resolve()),
    }
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(rows)} paired policy rows to {raw_path}")
    print(f"Saved manifest to {args.output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
