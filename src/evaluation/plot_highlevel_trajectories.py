from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.training.highlevel_env import HighLevelOptionEnv


OPTION_NAMES = ("pi1", "pi2", "pi3", "pi4")
LOW_MODEL_FILENAMES = (
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
)
SUMMARY_FIELDS = (
    "scenario",
    "mode",
    "episode_id",
    "outcome",
    "success",
    "switch_count",
    "option_sequence",
    "completed_phases",
)


def compressed_option_sequence(actions: list[int]) -> list[int]:
    sequence: list[int] = []
    for action in actions:
        if not sequence or sequence[-1] != action:
            sequence.append(action)
    return sequence


@dataclass
class TrajectoryRecorder:
    env: HighLevelOptionEnv
    evader_points: list[tuple[float, float, float]] = field(default_factory=list)
    pursuer_points: list[tuple[float, float, float]] = field(default_factory=list)
    option_switches: list[tuple[int, str]] = field(default_factory=list)
    phase_starts: list[tuple[int, str]] = field(default_factory=list)
    _original_inner_step: Callable[..., Any] | None = None

    def attach(self) -> None:
        if self._original_inner_step is not None:
            return
        self._original_inner_step = self.env.inner.step

        def recorded_step(action):
            assert self._original_inner_step is not None
            result = self._original_inner_step(action)
            self.append_state()
            return result

        self.env.inner.step = recorded_step

    def reset(self, phase_name: str) -> None:
        self.evader_points = []
        self.pursuer_points = []
        self.option_switches = []
        self.phase_starts = []
        self.append_state()
        self.phase_starts.append((0, phase_name))

    def append_state(self) -> int:
        state = self.env.inner.inner.state
        if state is None:
            raise RuntimeError("High-level environment state is missing while recording trajectory.")
        self.evader_points.append((state.evader.x, state.evader.y, state.evader.z))
        self.pursuer_points.append((state.pursuer.x, state.pursuer.y, state.pursuer.z))
        return len(self.evader_points) - 1

    def record_option(self, option: int, previous_option: int | None) -> None:
        if previous_option is None or option != previous_option:
            self.option_switches.append((len(self.evader_points) - 1, OPTION_NAMES[option]))

    def record_phase_transition(self, phase_name: str) -> None:
        index = self.append_state()
        self.phase_starts.append((index, phase_name))


@dataclass
class EpisodePlotData:
    scenario: str
    mode: str
    episode_id: int
    outcome: str
    success: bool
    switch_count: int
    option_sequence: list[int]
    completed_phases: int
    evader_points: list[tuple[float, float, float]]
    pursuer_points: list[tuple[float, float, float]]
    option_switches: list[tuple[int, str]]
    phase_starts: list[tuple[int, str]]

    def summary_row(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "success": self.success,
            "switch_count": self.switch_count,
            "option_sequence": "->".join(OPTION_NAMES[index] for index in self.option_sequence),
            "completed_phases": self.completed_phases,
        }


def rollout_episode(
    env: HighLevelOptionEnv,
    recorder: TrajectoryRecorder,
    *,
    episode_id: int,
    mode: str,
    fixed_policy: int,
    high_model: Any | None,
    scenario_set: str,
) -> EpisodePlotData:
    obs, info = env.reset(options={"scenario_set": scenario_set})
    scenario = str(info.get("scenario_name", "unknown"))
    recorder.reset(str(info.get("phase_name", "start")))
    actions: list[int] = []
    previous_option: int | None = None
    outcome = "timeout"

    while True:
        if mode == "fixed":
            option = fixed_policy
        else:
            assert high_model is not None
            prediction, _ = high_model.predict(obs, deterministic=True)
            option = int(prediction.item()) if hasattr(prediction, "item") else int(prediction)
        option = max(0, min(3, option))
        recorder.record_option(option, previous_option)
        actions.append(option)
        previous_option = option

        obs, _, terminated, truncated, info = env.step(option)
        outcome = str(info.get("outcome", "timeout" if truncated else "running"))
        if bool(info.get("phase_transition")) and not (terminated or truncated):
            recorder.record_phase_transition(str(info.get("phase_name", "phase")))
        if terminated or truncated:
            break

    return EpisodePlotData(
        scenario=scenario,
        mode=mode,
        episode_id=episode_id,
        outcome=outcome,
        success=outcome == "escaped",
        switch_count=int(info.get("switch_count", 0)),
        option_sequence=compressed_option_sequence(actions),
        completed_phases=int(info.get("completed_phases", 0)),
        evader_points=list(recorder.evader_points),
        pursuer_points=list(recorder.pursuer_points),
        option_switches=list(recorder.option_switches),
        phase_starts=list(recorder.phase_starts),
    )


def plot_priority(episode: EpisodePlotData, fixed_policy: int) -> tuple[int, int, int]:
    focus = int(episode.scenario == "sequential_rear_vertical_to_boundary")
    highlevel_success = int(episode.mode == "highlevel" and episode.success)
    fixed_pi3_failure = int(episode.mode == "fixed" and fixed_policy == 2 and not episode.success)
    return highlevel_success, fixed_pi3_failure, focus


def select_episodes_for_plot(
    episodes: list[EpisodePlotData],
    *,
    max_plots: int,
    fixed_policy: int,
    only_success: bool,
    only_failure: bool,
) -> list[EpisodePlotData]:
    filtered = [episode for episode in episodes if not only_success or episode.success]
    filtered = [episode for episode in filtered if not only_failure or not episode.success]
    return sorted(filtered, key=lambda episode: plot_priority(episode, fixed_policy), reverse=True)[:max_plots]


def save_plot(episode: EpisodePlotData, plot_dir: Path, bounds: tuple[float, float, float, float, float, float]) -> Path:
    import matplotlib.pyplot as plt

    evader = list(zip(*episode.evader_points))
    pursuer = list(zip(*episode.pursuer_points))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*evader, color="tab:blue", linewidth=2.0, label="evader")
    ax.plot(*pursuer, color="tab:red", linestyle="--", linewidth=1.6, label="pursuer")
    ax.scatter(*episode.evader_points[0], color="green", marker="o", s=65, label="evader start")
    ax.scatter(*episode.evader_points[-1], color="black", marker="X", s=70, label="evader end")

    for marker_index, phase_name in episode.phase_starts:
        x, y, z = episode.evader_points[marker_index]
        ax.scatter(x, y, z, color="tab:purple", marker="D", s=45)
        ax.text(x, y, z, f" phase:{phase_name}", color="tab:purple", fontsize=8)
    for marker_index, option_name in episode.option_switches:
        x, y, z = episode.evader_points[marker_index]
        ax.scatter(x, y, z, color="tab:orange", marker="^", s=45)
        ax.text(x, y, z, f" {option_name}", color="darkorange", fontsize=8)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"{episode.scenario} | mode={episode.mode} | outcome={episode.outcome} | "
        f"switch_count={episode.switch_count}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"highlevel_traj_ep{episode.episode_id:03d}_{episode.scenario}_{episode.mode}_{episode.outcome}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot high-level option trajectories for presentation")
    parser.add_argument("--mode", choices=["highlevel", "fixed"], default="highlevel")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential"], default="sequential")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fixed-policy", type=int, choices=range(4), default=2)
    parser.add_argument("--high-model", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    parser.add_argument("--max-plots", type=int, default=5)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--only-success", action="store_true")
    filter_group.add_argument("--only-failure", action="store_true")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_plots <= 0:
        parser.error("--max-plots must be positive")
    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib is not installed. Please install matplotlib to save high-level trajectory plots.")
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    low_paths = [checkpoint_dir / filename for filename in LOW_MODEL_FILENAMES]
    for path in low_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return
    high_path = Path(args.high_model)
    if args.mode == "highlevel" and not high_path.exists():
        print(f"Missing checkpoint: {high_path}")
        print("Please run this script locally after training high-level PPO selector.")
        return

    from stable_baselines3 import PPO, SAC

    low_models = [SAC.load(str(path)) for path in low_paths]
    high_model = PPO.load(str(high_path)) if args.mode == "highlevel" else None
    env = HighLevelOptionEnv(
        low_models=low_models,
        option_duration=args.option_duration,
        switch_penalty=args.switch_penalty,
        max_highlevel_steps=args.max_highlevel_steps,
        scenario_set=args.scenario_set,
    )
    recorder = TrajectoryRecorder(env)
    recorder.attach()
    episodes = [
        rollout_episode(
            env,
            recorder,
            episode_id=episode_id,
            mode=args.mode,
            fixed_policy=args.fixed_policy,
            high_model=high_model,
            scenario_set=args.scenario_set,
        )
        for episode_id in range(1, args.episodes + 1)
    ]
    selected = select_episodes_for_plot(
        episodes,
        max_plots=args.max_plots,
        fixed_policy=args.fixed_policy,
        only_success=args.only_success,
        only_failure=args.only_failure,
    )

    output_dir = Path(args.out_dir)
    plot_dir = output_dir / "highlevel_traj_plots"
    term_cfg = env.inner.inner.term_cfg
    bounds = (term_cfg.x_min, term_cfg.x_max, term_cfg.y_min, term_cfg.y_max, term_cfg.z_min, term_cfg.z_max)
    for episode in selected:
        print(f"Saved plot: {save_plot(episode, plot_dir, bounds)}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_dir / "highlevel_trajectory_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(episode.summary_row() for episode in selected)
    print(f"Saved summary CSV: {csv_path}")
    if not selected:
        print("No episodes matched the requested success/failure filter; summary CSV contains headers only.")


if __name__ == "__main__":
    main()
