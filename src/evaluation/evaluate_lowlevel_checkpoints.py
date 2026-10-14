"""Evaluate low-level checkpoints on paired initial states.

This is an evaluation-only diagnostic.  It does not train, change rewards, or
calculate ESAS/composite scores.  Every checkpoint is tested from the same
sampled initial state for each trial so that a mid-training reward dip can be
distinguished from a change in episode difficulty.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import statistics
from pathlib import Path
from typing import Any

def _checkpoint_paths(checkpoint_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = [Path(item) for item in requested]
    else:
        paths = sorted(checkpoint_dir.glob("pi2_flank_*_steps.zip"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint files found in {checkpoint_dir}")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint(s): {missing}")
    return paths


def _sample_initial_states(episodes: int, seed: int) -> list[Any]:
    from src.env.pursuit_escape_env import PursuitEscapeEnv

    states: list[Any] = []
    for trial_id in range(episodes):
        sampler = random.Random(seed + trial_id)
        env = PursuitEscapeEnv()
        env.reset(scenario="flank_threat", randomize=True, rng=sampler)
        if env.state is None:
            raise RuntimeError("environment did not produce an initial state")
        states.append(copy.deepcopy(env.state))
    return states


def _rollout_checkpoint(model: Any, initial_state: Any, trial_id: int) -> dict[str, Any]:
    from src.env.termination import TerminationState
    from src.training.sac_env import PursuitEscapeGymEnv

    env = PursuitEscapeGymEnv(scenario="flank_threat", randomize_reset=False)
    env.inner.state = copy.deepcopy(initial_state)
    env.inner.tstate = TerminationState()
    env.inner.current_scenario = "flank_threat"
    obs = env._flatten_obs(env.inner._observation(closing_speed=0.0))
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {"outcome": "timeout"}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if steps > 2_000:
            raise RuntimeError(f"trial {trial_id} exceeded the safety step limit")
    outcome = str(info.get("outcome", "timeout"))
    return {
        "trial_id": trial_id,
        "outcome": outcome,
        "total_reward": total_reward,
        "episode_length": steps,
    }


def _safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pi2 checkpoints on paired flank initial states")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for checkpoint evaluation") from exc

    checkpoint_dir = Path(args.checkpoint_dir)
    paths = _checkpoint_paths(checkpoint_dir, args.checkpoints)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "checkpoint_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_states = _sample_initial_states(args.episodes, args.seed)

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in paths:
        print(f"[checkpoint-eval] {checkpoint.name}")
        model = SAC.load(str(checkpoint), device=args.device)
        rows = [_rollout_checkpoint(model, state, trial_id) for trial_id, state in enumerate(initial_states)]
        for row in rows:
            raw_rows.append({"checkpoint": checkpoint.name, **row})
        rewards = [float(row["total_reward"]) for row in rows]
        lengths = [float(row["episode_length"]) for row in rows]
        summary_rows.append(
            {
                "checkpoint": checkpoint.name,
                "episodes": len(rows),
                "mean_reward": statistics.mean(rewards),
                "std_reward": _safe_std(rewards),
                "median_reward": statistics.median(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "mean_episode_length": statistics.mean(lengths),
                "success_rate": sum(row["outcome"] == "escaped" for row in rows) / len(rows),
                "capture_rate": sum(row["outcome"] == "captured" for row in rows) / len(rows),
                "out_of_bounds_rate": sum(row["outcome"] == "out_of_bounds" for row in rows) / len(rows),
                "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / len(rows),
            }
        )

    raw_path = output_dir / "pi2_checkpoint_eval_raw.csv"
    summary_path = output_dir / "pi2_checkpoint_eval_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote raw paired results to {raw_path}")
    print(f"Wrote checkpoint summary to {summary_path}")
    for row in summary_rows:
        print(
            f"{row['checkpoint']}: mean={row['mean_reward']:.2f}, "
            f"std={row['std_reward']:.2f}, success={row['success_rate']:.2f}, "
            f"capture={row['capture_rate']:.2f}, oob={row['out_of_bounds_rate']:.2f}, "
            f"timeout={row['timeout_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
