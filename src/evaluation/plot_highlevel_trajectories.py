from __future__ import annotations

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.training.highlevel_env import CONTINUOUS_SCENARIO_SETS, CONTINUOUS_SHOWCASE_SCENARIO, HighLevelOptionEnv


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
    "lowlevel_steps",
)


def compressed_option_sequence(actions: list[int]) -> list[int]:
    sequence: list[int] = []
    for action in actions:
        if not sequence or sequence[-1] != action:
            sequence.append(action)
    return sequence


def trajectory_segments(phase_starts: list[tuple[int, str]], point_count: int) -> list[tuple[int, int, str]]:
    if point_count <= 0:
        return []
    starts = sorted({max(0, min(point_count - 1, index)): name for index, name in phase_starts}.items())
    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, "phase"))
    segments: list[tuple[int, int, str]] = []
    for offset, (start_index, phase_name) in enumerate(starts):
        next_start = starts[offset + 1][0] if offset + 1 < len(starts) else point_count
        end_index = max(start_index, next_start - 1)
        segments.append((start_index, end_index, phase_name))
    return segments


def phase_reset_jumps(segments: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    return [(segments[index][1], segments[index + 1][0]) for index in range(len(segments) - 1)]


def sample_points(points: list[tuple[float, float, float]], sample_rate: int) -> list[tuple[float, float, float]]:
    if sample_rate <= 1 or len(points) <= 2:
        return points
    sampled = points[::sample_rate]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def boundary_priority_starts(points: list[int]) -> list[int]:
    starts: list[int] = []
    previous: int | None = None
    for index in sorted(set(points)):
        if previous is None or index > previous + 1:
            starts.append(index)
        previous = index
    return starts


def auto_plot_bounds(
    evader_points: list[tuple[float, float, float]],
    pursuer_points: list[tuple[float, float, float]],
    padding_ratio: float = 0.10,
) -> tuple[float, float, float, float, float, float]:
    points = evader_points + pursuer_points
    xs, ys, zs = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    xy_span = max(x_span, y_span)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    x_half = 0.5 * xy_span * (1.0 + padding_ratio)
    y_half = 0.5 * xy_span * (1.0 + padding_ratio)
    z_half = 0.5 * z_span * (1.0 + padding_ratio)
    return (
        x_center - x_half,
        x_center + x_half,
        y_center - y_half,
        y_center + y_half,
        z_center - z_half,
        z_center + z_half,
    )


@dataclass
class TrajectoryRecorder:
    env: HighLevelOptionEnv
    evader_points: list[tuple[float, float, float]] = field(default_factory=list)
    pursuer_points: list[tuple[float, float, float]] = field(default_factory=list)
    option_switches: list[tuple[int, str]] = field(default_factory=list)
    phase_starts: list[tuple[int, str]] = field(default_factory=list)
    regime_starts: list[tuple[int, str]] = field(default_factory=list)
    boundary_priority_points: list[int] = field(default_factory=list)
    _last_regime_name: str | None = None
    _original_inner_step: Callable[..., Any] | None = None
    _original_continuous_lowlevel_step: Callable[..., Any] | None = None

    def attach(self) -> None:
        if self._original_inner_step is not None:
            return
        self._original_inner_step = self.env.inner.step
        self._original_continuous_lowlevel_step = getattr(self.env, "_continuous_lowlevel_step", None)

        def recorded_step(action):
            assert self._original_inner_step is not None
            result = self._original_inner_step(action)
            self.append_state()
            return result

        def recorded_continuous_step(action, regime):
            assert self._original_continuous_lowlevel_step is not None
            result = self._original_continuous_lowlevel_step(action, regime)
            index = self.append_state()
            self.record_regime(regime, index=index)
            if bool(getattr(self.env, "continuous_last_regime_info", {}).get("boundary_priority_active", False)):
                self.boundary_priority_points.append(index)
            return result

        self.env.inner.step = recorded_step
        if self._original_continuous_lowlevel_step is not None:
            self.env._continuous_lowlevel_step = recorded_continuous_step

    def reset(self, phase_name: str) -> None:
        self.evader_points = []
        self.pursuer_points = []
        self.option_switches = []
        self.phase_starts = []
        self.regime_starts = []
        self.boundary_priority_points = []
        self._last_regime_name = None
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

    def record_regime(self, regime_name: str, *, index: int | None = None) -> None:
        if self._last_regime_name == regime_name:
            return
        marker_index = len(self.evader_points) - 1 if index is None else index
        self.regime_starts.append((marker_index, regime_name))
        self._last_regime_name = regime_name


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
    regime_starts: list[tuple[int, str]]
    boundary_priority_points: list[int]
    lowlevel_steps: int

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
            "lowlevel_steps": self.lowlevel_steps,
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
    scenario_name: str | None = None,
) -> EpisodePlotData:
    reset_options = {"scenario_set": scenario_set}
    if scenario_name:
        reset_options["scenario_name"] = scenario_name
    obs, info = env.reset(options=reset_options)
    scenario = str(info.get("scenario_name", "unknown"))
    recorder.reset(str(info.get("phase_name", "start")))
    if scenario_set in CONTINUOUS_SCENARIO_SETS:
        recorder.record_regime(str(info.get("regime_name", "start")), index=0)
    actions: list[int] = []
    previous_option: int | None = None
    outcome = "timeout"

    while True:
        if mode == "fixed":
            option = fixed_policy
        elif mode == "random":
            option = random.randint(0, 3)
        elif mode == "regime_oracle":
            option = {"rear": 0, "flank": 1, "boundary": 2, "vertical": 3}.get(str(info.get("regime_name", "rear")), 2)
        elif mode == "continuous_heuristic":
            regime = str(info.get("regime_name", "rear"))
            min_margin = float(info.get("min_boundary_margin", 1.0))
            boundary_enter = float(info.get("boundary_priority_enter", 0.24))
            boundary_active = bool(info.get("boundary_priority_active", False))
            if min_margin <= boundary_enter or boundary_active or regime == "boundary":
                option = 2
            elif regime == "flank":
                option = 1
            elif regime == "vertical":
                option = 3
            elif regime == "rear":
                option = 0
            else:
                option = 2
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
        regime_starts=list(recorder.regime_starts),
        boundary_priority_points=list(recorder.boundary_priority_points),
        lowlevel_steps=int(info.get("continuous_lowlevel_steps", 0)),
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
    one_per_scenario: bool = False,
    one_per_option_sequence: bool = False,
) -> list[EpisodePlotData]:
    filtered = [episode for episode in episodes if not only_success or episode.success]
    filtered = [episode for episode in filtered if not only_failure or not episode.success]
    selected: list[EpisodePlotData] = []
    seen_scenarios: set[str] = set()
    seen_sequences: set[tuple[int, ...]] = set()
    for episode in sorted(filtered, key=lambda item: plot_priority(item, fixed_policy), reverse=True):
        sequence = tuple(episode.option_sequence)
        if one_per_scenario and episode.scenario in seen_scenarios:
            continue
        if one_per_option_sequence and sequence in seen_sequences:
            continue
        selected.append(episode)
        seen_scenarios.add(episode.scenario)
        seen_sequences.add(sequence)
        if len(selected) >= max_plots:
            break
    return selected

def _plot_segment(
    ax,
    points: list[tuple[float, float, float]],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None,
    alpha: float = 1.0,
    view: str = "3d",
) -> None:
    if len(points) == 1:
        if view == "topdown":
            ax.scatter(points[0][0], points[0][1], color=color, marker=".", s=25, label=label, alpha=alpha)
        else:
            ax.scatter(*points[0], color=color, marker=".", s=25, label=label, alpha=alpha)
        return
    if view == "topdown":
        xs, ys, _ = zip(*points)
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)
    else:
        coords = list(zip(*points))
        ax.plot(*coords, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)


def _scatter_point(ax, point: tuple[float, float, float], *, view: str, **kwargs) -> None:
    if view == "topdown":
        ax.scatter(point[0], point[1], **kwargs)
    else:
        ax.scatter(*point, **kwargs)


def _annotate_point(ax, point: tuple[float, float, float], text: str, *, view: str, **kwargs) -> None:
    if view == "topdown":
        ax.text(point[0], point[1], text, **kwargs)
    else:
        ax.text(*point, text, **kwargs)


def save_plot(
    episode: EpisodePlotData,
    plot_dir: Path,
    bounds: tuple[float, float, float, float, float, float],
    *,
    break_at_phase_transition: bool = True,
    show_phase_reset_jump: bool = False,
    showcase_mode: str = "phase_based",
    plot_sample_rate: int = 5,
    max_annotations: int = 8,
    no_text_annotations: bool = False,
    view: str = "3d",
) -> Path:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=None if view == "topdown" else "3d")
    is_continuous = episode.scenario in CONTINUOUS_SCENARIO_SETS
    segments = (
        trajectory_segments(episode.phase_starts, len(episode.evader_points))
        if break_at_phase_transition and not is_continuous
        else [(0, len(episode.evader_points) - 1, "all")]
    )
    for segment_index, (start, end, _) in enumerate(segments):
        evader_segment = episode.evader_points[start : end + 1]
        pursuer_segment = sample_points(episode.pursuer_points[start : end + 1], plot_sample_rate)
        _plot_segment(
            ax,
            evader_segment,
            color="tab:blue",
            linestyle="-",
            linewidth=2.6,
            label="evader" if segment_index == 0 else None,
            view=view,
        )
        _plot_segment(
            ax,
            pursuer_segment,
            color="tab:red",
            linestyle="--",
            linewidth=1.4,
            label=f"pursuer (sample/{max(plot_sample_rate, 1)})" if segment_index == 0 else None,
            alpha=0.45,
            view=view,
        )

    if break_at_phase_transition and show_phase_reset_jump and not is_continuous:
        for jump_index, (end_index, start_index) in enumerate(phase_reset_jumps(segments)):
            evader_jump = [episode.evader_points[end_index], episode.evader_points[start_index]]
            pursuer_jump = [episode.pursuer_points[end_index], episode.pursuer_points[start_index]]
            _plot_segment(
                ax,
                evader_jump,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label="phase reset jump (not physical)" if jump_index == 0 else None,
                view=view,
            )
            _plot_segment(ax, pursuer_jump, color="gray", linestyle=":", linewidth=1.0, label=None, view=view)

    _scatter_point(ax, episode.evader_points[0], view=view, color="green", marker="o", s=75, label="evader start", zorder=5)
    _scatter_point(ax, episode.evader_points[-1], view=view, color="black", marker="X", s=85, label="evader end", zorder=5)

    annotation_count = 0
    option_styles = {
        "pi1": ("tab:orange", "o"),
        "pi2": ("tab:purple", "s"),
        "pi3": ("tab:brown", "^"),
        "pi4": ("tab:pink", "v"),
    }

    for marker_index, phase_name in ([] if is_continuous else episode.phase_starts):
        x, y, z = episode.evader_points[marker_index]
        point = (x, y, z)
        _scatter_point(ax, point, view=view, color="tab:purple", marker="D", s=45, label="phase start" if marker_index == episode.phase_starts[0][0] else None)
        if not no_text_annotations and annotation_count < max_annotations:
            _annotate_point(ax, point, f" phase:{phase_name}", view=view, color="tab:purple", fontsize=8)
            annotation_count += 1
    seen_option_names: set[str] = set()
    for marker_index, option_name in episode.option_switches:
        point = episode.evader_points[marker_index]
        color, marker = option_styles.get(option_name, ("tab:orange", "o"))
        label = f"option switch {option_name}" if option_name not in seen_option_names else None
        seen_option_names.add(option_name)
        _scatter_point(ax, point, view=view, color=color, marker=marker, s=34, alpha=0.85, label=label, zorder=6)
    for regime_offset, (marker_index, regime_name) in enumerate(episode.regime_starts):
        point = episode.evader_points[marker_index]
        _scatter_point(ax, point, view=view, color="tab:cyan", marker="s", s=38, alpha=0.80, label="regime switch" if regime_offset == 0 else None, zorder=5)
        if not no_text_annotations and annotation_count < max_annotations:
            _annotate_point(ax, point, f" {regime_name}", view=view, color="teal", fontsize=8)
            annotation_count += 1
    for boundary_offset, marker_index in enumerate(boundary_priority_starts(episode.boundary_priority_points)):
        point = episode.evader_points[marker_index]
        _scatter_point(
            ax,
            point,
            view=view,
            color="darkorange",
            marker="*",
            s=80,
            alpha=0.95,
            label="boundary priority start" if boundary_offset == 0 else None,
            zorder=7,
        )

    x_min, x_max, y_min, y_max, z_min, z_max = auto_plot_bounds(episode.evader_points, episode.pursuer_points)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if view != "topdown":
        ax.set_zlim(z_min, z_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if view == "topdown":
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_zlabel("z")
    option_sequence = "->".join(OPTION_NAMES[index] for index in episode.option_sequence)
    if is_continuous:
        first_line = f"{episode.scenario} | mode={episode.mode} | outcome={episode.outcome} | lowlevel_steps={episode.lowlevel_steps}"
    else:
        title_suffix = "phase-based sequential rollout" if showcase_mode == "phase_based" else "continuous showcase requested (not benchmark)"
        first_line = f"{episode.scenario} | mode={episode.mode} | outcome={episode.outcome} | {title_suffix}"
    ax.set_title(f"{first_line}\noption_sequence={option_sequence} | switch_count={episode.switch_count}")
    ax.legend(loc="upper left")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    view_suffix = "_topdown" if view == "topdown" else ""
    output_path = plot_dir / f"highlevel_traj_ep{episode.episode_id:03d}_{episode.scenario}_{episode.mode}_{episode.outcome}{view_suffix}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot high-level option trajectories for presentation")
    parser.add_argument("--mode", choices=["highlevel", "fixed", "random", "continuous_heuristic", "regime_oracle"], default="highlevel")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential", "continuous_pursuit", "continuous_showcase"], default="sequential")
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fixed-policy", type=int, choices=range(4), default=2)
    parser.add_argument("--high-model", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    parser.add_argument("--max-plots", type=int, default=5)
    parser.add_argument("--one-per-scenario", action="store_true")
    parser.add_argument("--one-per-option-sequence", action="store_true")
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--only-success", action="store_true")
    filter_group.add_argument("--only-failure", action="store_true")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    parser.add_argument("--break-at-phase-transition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-phase-reset-jump", action="store_true")
    parser.add_argument("--showcase-mode", choices=["phase_based", "continuous"], default="phase_based")
    parser.add_argument("--episode-lowlevel-steps", type=int, default=400)
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.20)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    parser.add_argument("--min-regime-hold-steps", type=int, default=20)
    parser.add_argument("--boundary-priority-enter", type=float, default=0.24)
    parser.add_argument("--boundary-priority-exit", type=float, default=0.32)
    parser.add_argument("--showcase-bound-scale", type=float, default=2.5)
    parser.add_argument("--showcase-z-bound-scale", type=float, default=1.5)
    parser.add_argument("--plot-sample-rate", type=int, default=5)
    parser.add_argument("--max-annotations", type=int, default=8)
    parser.add_argument("--no-text-annotations", action="store_true")
    parser.add_argument("--view", choices=["3d", "topdown"], default="3d")
    args = parser.parse_args()
    if args.scenario_set == CONTINUOUS_SHOWCASE_SCENARIO and args.episode_lowlevel_steps == 400:
        args.episode_lowlevel_steps = 500

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_plots <= 0:
        parser.error("--max-plots must be positive")
    if args.plot_sample_rate <= 0:
        parser.error("--plot-sample-rate must be positive")
    if args.max_annotations < 0:
        parser.error("--max-annotations must be non-negative")
    if args.showcase_mode == "continuous":
        print("showcase-mode=continuous only changes plot labeling; benchmark dynamics are selected by --scenario-set.")
    if args.mode in {"continuous_heuristic", "regime_oracle"} and args.scenario_set not in CONTINUOUS_SCENARIO_SETS:
        parser.error(f"--mode {args.mode} requires --scenario-set continuous_pursuit or continuous_showcase")
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
        episode_lowlevel_steps=args.episode_lowlevel_steps,
        regime_duration=args.regime_duration,
        pursuer_speed_ratio=args.pursuer_speed_ratio,
        regime_schedule=args.regime_schedule,
        min_regime_hold_steps=args.min_regime_hold_steps,
        boundary_priority_enter=args.boundary_priority_enter,
        boundary_priority_exit=args.boundary_priority_exit,
        showcase_bound_scale=args.showcase_bound_scale,
        showcase_z_bound_scale=args.showcase_z_bound_scale,
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
            scenario_name=args.scenario_name,
        )
        for episode_id in range(1, args.episodes + 1)
    ]
    selected = select_episodes_for_plot(
        episodes,
        max_plots=args.max_plots,
        fixed_policy=args.fixed_policy,
        only_success=args.only_success,
        only_failure=args.only_failure,
        one_per_scenario=args.one_per_scenario,
        one_per_option_sequence=args.one_per_option_sequence,
    )

    output_dir = Path(args.out_dir)
    plot_dir = output_dir / "highlevel_traj_plots"
    term_cfg = env.inner.inner.term_cfg
    bounds = (term_cfg.x_min, term_cfg.x_max, term_cfg.y_min, term_cfg.y_max, term_cfg.z_min, term_cfg.z_max)
    for episode in selected:
        print(
            f"Saved plot: {save_plot(episode, plot_dir, bounds, break_at_phase_transition=args.break_at_phase_transition, show_phase_reset_jump=args.show_phase_reset_jump, showcase_mode=args.showcase_mode, plot_sample_rate=args.plot_sample_rate, max_annotations=args.max_annotations, no_text_annotations=args.no_text_annotations, view=args.view)}"
        )

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
