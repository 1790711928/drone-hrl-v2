"""Reusable training logging and manifest helpers.

This module only records training metadata and episode statistics.  It does not
change environments, rewards, observations, actions, or algorithm networks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:  # Allows static CLI/help inspection without SB3 installed.
    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose
            self.locals = {}
            self.num_timesteps = 0


def _configure_sb3(*args: Any, **kwargs: Any):
    from stable_baselines3.common.logger import configure

    return configure(*args, **kwargs)

PROTECTED_OUTPUT_PREFIXES = (
    Path("outputs/checkpoints"),
    Path("outputs/evaluation"),
    Path("outputs/paper_eval_core"),
    Path("outputs/paper_audit"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.resolve()


def assert_new_output_dir(path: Path) -> None:
    """Reject all writes under directories containing legacy paper outputs."""
    candidate = _relative(path)
    for protected in PROTECTED_OUTPUT_PREFIXES:
        protected_relative = _relative(protected)
        if candidate == protected_relative or protected_relative in candidate.parents:
            raise ValueError(f"new training output cannot be written under protected path: {path}")


def configure_logger(log_dir: Path, tensorboard_dir: Path | None = None):
    """Configure stdout/CSV progress and a separate TensorBoard directory."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_sb3(str(log_dir), ["stdout", "csv"])
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_logger = _configure_sb3(str(tensorboard_dir), ["tensorboard"])
        logger.output_formats.extend(tensorboard_logger.output_formats)
    return logger


class EpisodeRewardCallback(BaseCallback):
    """Persist Monitor episode records with explicit policy/run metadata."""

    FIELDNAMES = [
        "episode_index",
        "global_step",
        "episode_return",
        "episode_length",
        "outcome",
        "escaped",
        "early_safe_escape",
        "disabled_policy",
        "disabled_policy_index",
        "action_remap_count",
        "policy_name",
        "seed",
        "elapsed_time",
    ]

    def __init__(self, output_path: Path, policy_name: str, seed: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.output_path = output_path
        self.policy_name = policy_name
        self.seed = seed
        self.episode_count = 0
        self.nan_or_inf = False
        self.started_at = 0.0
        self._handle = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self.started_at = time.monotonic()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            for reward in rewards:
                if not math.isfinite(float(reward)):
                    self.nan_or_inf = True
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if not episode:
                continue
            episode_return = float(episode["r"])
            episode_length = int(episode["l"])
            outcome = str(info.get("outcome", "")) if isinstance(info, dict) else ""
            if not math.isfinite(episode_return):
                self.nan_or_inf = True
            self.episode_count += 1
            self._writer.writerow(
                {
                    "episode_index": self.episode_count,
                    "global_step": int(self.num_timesteps),
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "outcome": outcome,
                    "escaped": int(outcome == "escaped"),
                    "early_safe_escape": int(bool(info.get("early_safe_escape", False)))
                    if isinstance(info, dict)
                    else 0,
                    "disabled_policy": str(info.get("disabled_policy", "none"))
                    if isinstance(info, dict)
                    else "none",
                    "disabled_policy_index": int(info.get("disabled_policy_index", -1))
                    if isinstance(info, dict)
                    else -1,
                    "action_remap_count": int(info.get("action_remap_count", 0))
                    if isinstance(info, dict)
                    else 0,
                    "policy_name": self.policy_name,
                    "seed": self.seed,
                    "elapsed_time": time.monotonic() - self.started_at,
                }
            )
            self._handle.flush()
        return True

    def _on_training_end(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()


def checkpoint_callback(checkpoint_dir: Path, save_freq: int, policy_name: str):
    if save_freq <= 0:
        return None
    from stable_baselines3.common.callbacks import CheckpointCallback

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=policy_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def combine_callbacks(callbacks: Sequence[Any]):
    callbacks = [callback for callback in callbacks if callback is not None]
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    from stable_baselines3.common.callbacks import CallbackList

    return CallbackList(callbacks)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(env: Any) -> dict[str, Any]:
    return {
        "environment_name": type(env).__name__,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def build_run_manifest(
    *,
    experiment_name: str,
    policy_name: str,
    status: str,
    seed: int,
    total_timesteps: int,
    env: Any,
    algorithm: str,
    hyperparameters: dict[str, Any],
    run_dir: Path,
    final_model: Path,
    reward_csv: Path,
    monitor_csv: Path,
    tensorboard_dir: Path,
    checkpoint_dir: Path,
    started_at: str,
    finished_at: str | None,
    callback: EpisodeRewardCallback,
    actual_num_timesteps: int,
) -> dict[str, Any]:
    reward_path = Path("src/env/reward.py")
    env_path = Path("src/env/pursuit_escape_env.py")
    return {
        "experiment_name": experiment_name,
        "policy_name": policy_name,
        "status": status,
        "date": utc_now(),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "actual_num_timesteps": actual_num_timesteps,
        **environment_metadata(env),
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "reward_version": file_sha256(reward_path) if reward_path.exists() else "unknown",
        "environment_version": file_sha256(env_path) if env_path.exists() else "unknown",
        "code_commit": git_commit(),
        "training_started_at": started_at,
        "training_finished_at": finished_at,
        "run_dir": str(run_dir.resolve()),
        "final_model_path": str(final_model.resolve()),
        "reward_csv_path": str(reward_csv.resolve()),
        "monitor_csv_path": str(monitor_csv.resolve()),
        "tensorboard_path": str(tensorboard_dir.resolve()),
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "episode_count": callback.episode_count,
        "nan_or_inf_detected": callback.nan_or_inf,
    }


def build_experiment_manifest(
    *,
    experiment_name: str,
    status: str,
    seed: int,
    total_timesteps: int,
    policy_runs: list[dict[str, Any]],
    started_at: str,
    finished_at: str | None,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "status": status,
        "date": utc_now(),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_started_at": started_at,
        "training_finished_at": finished_at,
        "code_commit": git_commit(),
        "policy_runs": policy_runs,
    }
