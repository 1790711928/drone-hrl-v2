"""Plot high-level option trajectories from a conflict-marker-free clean file."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.training.highlevel_env import CONTINUOUS_SCENARIO_SETS, CONTINUOUS_SHOWCASE_SCENARIO, SCRIPTED_SHOWCASE_SCENARIO, HighLevelOptionEnv


OPTION_NAMES = ("pi1", "pi2", "pi3", "pi4")
STRATEGY_LABELS = {
    0: "Rear strategy",
    1: "Flank strategy",
    2: "Boundary strategy",
    3: "Vertical strategy",
}
OPTION_TO_STRATEGY = {name: STRATEGY_LABELS[index] for index, name in enumerate(OPTION_NAMES)}
SCRIPTED_SHOWCASE_SCRIPT_NAME = "rear60-flank60-vertical60-boundary60-rear80-flank80-vertical100"
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
    "unique_option_count",
    "option_sequence",
    "actual_regime_sequence",
    "showcase_script",
    "showcase_seed",
    "completed_phases",
    "lowlevel_steps",
)


def compressed_option_sequence(actions: list[int]) -> list[int]:
    sequence: list[int] = []
    for action in actions:
        if not sequence or sequence[-1] != action:
            sequence.append(action)
    return sequence


def compressed_name_sequence(names: list[str]) -> list[str]:
    sequence: list[str] = []
    for name in names:
        if not sequence or sequence[-1] != name:
            sequence.append(name)
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


def option_trajectory_segments(option_switches: list[tuple[int, str]], point_count: int) -> list[tuple[int, int, str]]:
    if point_count <= 0:
        return []
    starts = sorted((max(0, min(point_count - 1, index)), option_name) for index, option_name in option_switches)
    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, "pi1"))
    segments: list[tuple[int, int, str]] = []
    for offset, (start, option_name) in enumerate(starts):
        next_start = starts[offset + 1][0] if offset + 1 < len(starts) else point_count
        end = max(start, next_start - 1)
        segments.append((start, end, option_name))
    return segments


def pursuit_link_indices(point_count: int, interval: int) -> list[int]:
    if point_count <= 1 or interval <= 0:
        return []
    indices = list(range(0, point_count, interval))
    if indices[-1] != point_count - 1:
        indices.append(point_count - 1)
    return indices


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


def equalize_3d_bounds(bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
    return (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
        z_center - half_span,
        z_center + half_span,
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
    showcase_seed: int | str = ""

    def summary_row(self) -> dict[str, Any]:
        option_sequence = "->".join(OPTION_NAMES[index] for index in self.option_sequence)
        actual_regime_sequence = "->".join(compressed_name_sequence([name for _, name in self.regime_starts]))
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "success": self.success,
            "switch_count": self.switch_count,
            "unique_option_count": len(set(self.option_sequence)),
            "option_sequence": option_sequence,
            "actual_regime_sequence": actual_regime_sequence,
            "showcase_script": SCRIPTED_SHOWCASE_SCRIPT_NAME if self.scenario == SCRIPTED_SHOWCASE_SCENARIO or self.mode == "scripted_showcase" else "",
            "showcase_seed": self.showcase_seed,
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
    reset_seed = episode_id if scenario_set == SCRIPTED_SHOWCASE_SCENARIO else None
    obs, info = env.reset(seed=reset_seed, options=reset_options)
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
        elif mode in {"regime_oracle", "scripted_showcase"}:
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
        showcase_seed=info.get("showcase_seed", ""),
    )


def plot_priority(episode: EpisodePlotData, fixed_policy: int) -> tuple[int, int, int, int, int, int]:
    focus = int(episode.scenario == "sequential_rear_vertical_to_boundary")
    highlevel_success = int(episode.mode == "highlevel" and episode.success)
    fixed_pi3_failure = int(episode.mode == "fixed" and fixed_policy == 2 and not episode.success)
    return (
        highlevel_success,
        fixed_pi3_failure,
        focus,
        len(set(episode.option_sequence)),
        episode.switch_count,
        episode.lowlevel_steps,
    )


def select_episodes_for_plot(
    episodes: list[EpisodePlotData],
    *,
    max_plots: int,
    fixed_policy: int,
    only_success: bool,
    only_failure: bool,
    min_switch_count: int = 0,
    min_lowlevel_steps: int = 0,
    min_unique_options: int = 1,
    one_per_scenario: bool = False,
    one_per_option_sequence: bool = False,
) -> list[EpisodePlotData]:
    filtered = [episode for episode in episodes if not only_success or episode.success]
    filtered = [episode for episode in filtered if not only_failure or not episode.success]
    filtered = [episode for episode in filtered if episode.switch_count >= min_switch_count]
    filtered = [episode for episode in filtered if episode.lowlevel_steps >= min_lowlevel_steps]
    filtered = [episode for episode in filtered if len(set(episode.option_sequence)) >= min_unique_options]
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
    show_callouts: bool = False,
    show_strategy_switch_markers: bool = False,
    show_regime_switch_markers: bool = False,
    show_boundary_priority_markers: bool = False,
    show_pursuit_links: bool = False,
    pursuit_link_interval: int = 60,
    view: str = "3d",
) -> Path:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=None if view == "topdown" else "3d")
    is_continuous = episode.scenario in CONTINUOUS_SCENARIO_SETS
    is_scripted_showcase = episode.scenario == SCRIPTED_SHOWCASE_SCENARIO or episode.mode == "scripted_showcase"
    suppress_text_annotations = no_text_annotations or (is_scripted_showcase and not show_callouts)
    draw_strategy_switch_markers = show_strategy_switch_markers or not is_scripted_showcase
    draw_regime_switch_markers = show_regime_switch_markers or not is_scripted_showcase
    draw_boundary_priority_markers = show_boundary_priority_markers or not is_scripted_showcase
    segments = (
        trajectory_segments(episode.phase_starts, len(episode.evader_points))
        if break_at_phase_transition and not is_continuous
        else [(0, len(episode.evader_points) - 1, "all")]
    )
    for segment_index, (start, end, _) in enumerate(segments):
        evader_segment = episode.evader_points[start : end + 1]
        pursuer_segment = sample_points(episode.pursuer_points[start : end + 1], plot_sample_rate)
        if is_scripted_showcase:
            # Scripted showcase colors the evader trajectory by the active strategy,
            # making the executed option intervals visible without in-plot text.
            pass
        else:
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
            label=("pursuer" if is_scripted_showcase else f"pursuer (sample/{max(plot_sample_rate, 1)})") if segment_index == 0 else None,
            alpha=0.45,
            view=view,
        )

    if is_scripted_showcase:
        seen_strategy_labels: set[str] = set()
        option_styles_for_lines = {
            "pi1": ("tab:orange", "Rear strategy"),
            "pi2": ("tab:purple", "Flank strategy"),
            "pi3": ("tab:brown", "Boundary strategy"),
            "pi4": ("tab:pink", "Vertical strategy"),
        }
        for start, end, option_name in option_trajectory_segments(episode.option_switches, len(episode.evader_points)):
            color, label = option_styles_for_lines.get(option_name, ("tab:blue", option_name))
            segment_points = episode.evader_points[start : end + 1]
            if len(segment_points) < 2:
                continue
            _plot_segment(
                ax,
                segment_points,
                color=color,
                linestyle="-",
                linewidth=2.8,
                label=label if label not in seen_strategy_labels else None,
                view=view,
            )
            seen_strategy_labels.add(label)

    if show_pursuit_links and is_scripted_showcase:
        for link_offset, index in enumerate(pursuit_link_indices(len(episode.evader_points), pursuit_link_interval)):
            _plot_segment(
                ax,
                [episode.evader_points[index], episode.pursuer_points[index]],
                color="0.55",
                linestyle="-",
                linewidth=0.7,
                label="pursuit link" if link_offset == 0 else None,
                alpha=0.28,
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
        if not suppress_text_annotations and annotation_count < max_annotations:
            _annotate_point(ax, point, f" phase:{phase_name}", view=view, color="tab:purple", fontsize=8)
            annotation_count += 1
    seen_option_names: set[str] = set()
    if draw_strategy_switch_markers:
        for marker_index, option_name in episode.option_switches:
            point = episode.evader_points[marker_index]
            color, marker = option_styles.get(option_name, ("tab:orange", "o"))
            if episode.scenario == SCRIPTED_SHOWCASE_SCENARIO or episode.mode == "scripted_showcase":
                label = OPTION_TO_STRATEGY.get(option_name, option_name) if option_name not in seen_option_names else None
            else:
                label = f"option switch {option_name}" if option_name not in seen_option_names else None
            seen_option_names.add(option_name)
            marker_size = 20 if is_scripted_showcase else 34
            _scatter_point(ax, point, view=view, color=color, marker=marker, s=marker_size, alpha=0.75, label=label, zorder=6)

    if show_callouts and is_scripted_showcase:
        for callout_index, (marker_index, option_name) in enumerate(episode.option_switches[: max(0, min(max_annotations, 6))]):
            point = episode.evader_points[marker_index]
            color, _ = option_styles.get(option_name, ("tab:orange", "o"))
            label_text = OPTION_TO_STRATEGY.get(option_name, option_name).replace(" strategy", "")
            dx = 5.0 + 1.5 * (callout_index % 2)
            dy = 4.0 * (1 if callout_index % 2 == 0 else -1)
            dz = 3.0 if view != "topdown" else 0.0
            label_point = (point[0] + dx, point[1] + dy, point[2] + dz)
            _plot_segment(ax, [point, label_point], color=color, linestyle=":", linewidth=0.8, label=None, alpha=0.65, view=view)
            _annotate_point(ax, label_point, label_text, view=view, color=color, fontsize=8)

    if draw_regime_switch_markers:
        for regime_offset, (marker_index, regime_name) in enumerate(episode.regime_starts):
            point = episode.evader_points[marker_index]
            _scatter_point(ax, point, view=view, color="tab:cyan", marker="s", s=38, alpha=0.80, label="regime switch" if regime_offset == 0 else None, zorder=5)
            if not suppress_text_annotations and annotation_count < max_annotations:
                _annotate_point(ax, point, f" {regime_name}", view=view, color="teal", fontsize=8)
                annotation_count += 1
    if draw_boundary_priority_markers:
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

    bounds = auto_plot_bounds(episode.evader_points, episode.pursuer_points)
    if view != "topdown":
        bounds = equalize_3d_bounds(bounds)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if view != "topdown":
        ax.set_zlim(z_min, z_max)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if view == "topdown":
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_zlabel("z")
    if episode.scenario == SCRIPTED_SHOWCASE_SCENARIO or episode.mode == "scripted_showcase":
        option_sequence = "->".join(STRATEGY_LABELS[index] for index in episode.option_sequence)
    else:
        option_sequence = "->".join(OPTION_NAMES[index] for index in episode.option_sequence)
    if is_scripted_showcase:
        first_line = f"Scripted showcase rollout | outcome={episode.outcome} | lowlevel_steps={episode.lowlevel_steps}"
        second_line = f"switch_count={episode.switch_count} | unique_strategies={len(set(episode.option_sequence))}"
    elif is_continuous:
        first_line = f"{episode.scenario} | mode={episode.mode} | outcome={episode.outcome} | lowlevel_steps={episode.lowlevel_steps}"
        second_line = f"option_sequence={option_sequence} | switch_count={episode.switch_count}"
    else:
        title_suffix = "phase-based sequential rollout" if showcase_mode == "phase_based" else "continuous showcase requested (not benchmark)"
        first_line = f"{episode.scenario} | mode={episode.mode} | outcome={episode.outcome} | {title_suffix}"
        second_line = f"option_sequence={option_sequence} | switch_count={episode.switch_count}"
    ax.set_title(f"{first_line}\n{second_line}")
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
    parser.add_argument("--mode", choices=["highlevel", "fixed", "random", "continuous_heuristic", "regime_oracle", "scripted_showcase"], default="highlevel")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential", "continuous_pursuit", "continuous_showcase", "scripted_showcase"], default="sequential")
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
    parser.add_argument("--show-callouts", action="store_true")
    parser.add_argument("--show-strategy-switch-markers", action="store_true")
    parser.add_argument("--show-regime-switch-markers", action="store_true")
    parser.add_argument("--show-boundary-priority-markers", action="store_true")
    parser.add_argument("--show-pursuit-links", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pursuit-link-interval", type=int, default=60)
    parser.add_argument("--view", choices=["3d", "topdown"], default="3d")
    parser.add_argument("--min-switch-count", type=int, default=0)
    parser.add_argument("--min-lowlevel-steps", type=int, default=0)
    parser.add_argument("--min-unique-options", type=int, default=1)
    parser.add_argument("--max-rollout-attempts", type=int, default=0)
    args = parser.parse_args()
    if args.scenario_set in {CONTINUOUS_SHOWCASE_SCENARIO, SCRIPTED_SHOWCASE_SCENARIO} and args.episode_lowlevel_steps == 400:
        args.episode_lowlevel_steps = 500

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_plots <= 0:
        parser.error("--max-plots must be positive")
    if args.plot_sample_rate <= 0:
        parser.error("--plot-sample-rate must be positive")
    if args.max_annotations < 0:
        parser.error("--max-annotations must be non-negative")
    if args.min_switch_count < 0 or args.min_lowlevel_steps < 0 or args.min_unique_options < 1:
        parser.error("episode filter thresholds must be non-negative and --min-unique-options must be >= 1")
    if args.pursuit_link_interval <= 0:
        parser.error("--pursuit-link-interval must be positive")
    if args.showcase_mode == "continuous":
        print("showcase-mode=continuous only changes plot labeling; benchmark dynamics are selected by --scenario-set.")
    if args.mode in {"continuous_heuristic", "regime_oracle", "scripted_showcase"} and args.scenario_set not in CONTINUOUS_SCENARIO_SETS:
        parser.error(f"--mode {args.mode} requires a continuous scenario set")
    if args.mode == "scripted_showcase" and args.scenario_set != SCRIPTED_SHOWCASE_SCENARIO:
        parser.error("--mode scripted_showcase requires --scenario-set scripted_showcase")
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
    max_attempts = args.max_rollout_attempts or max(args.episodes, args.max_plots * 20)
    episodes: list[EpisodePlotData] = []
    selected: list[EpisodePlotData] = []
    for episode_id in range(1, max_attempts + 1):
        episode = rollout_episode(
            env,
            recorder,
            episode_id=episode_id,
            mode=args.mode,
            fixed_policy=args.fixed_policy,
            high_model=high_model,
            scenario_set=args.scenario_set,
            scenario_name=args.scenario_name,
        )
        episodes.append(episode)
        selected = select_episodes_for_plot(
            episodes,
            max_plots=args.max_plots,
            fixed_policy=args.fixed_policy,
            only_success=args.only_success,
            only_failure=args.only_failure,
            min_switch_count=args.min_switch_count,
            min_lowlevel_steps=args.min_lowlevel_steps,
            min_unique_options=args.min_unique_options,
            one_per_scenario=args.one_per_scenario,
            one_per_option_sequence=args.one_per_option_sequence,
        )
        if len(selected) >= args.max_plots and episode_id >= args.episodes:
            break
        if episode_id >= args.episodes and not any(
            (args.min_switch_count, args.min_lowlevel_steps, args.min_unique_options > 1)
        ):
            break
    if not selected:
        print(
            "No episodes matched plot filters: "
            f"min_switch_count={args.min_switch_count}, "
            f"min_lowlevel_steps={args.min_lowlevel_steps}, "
            f"min_unique_options={args.min_unique_options}. "
            f"Attempts={len(episodes)}."
        )

    output_dir = Path(args.out_dir)
    plot_dir = output_dir / "highlevel_traj_plots"
    term_cfg = env.inner.inner.term_cfg
    bounds = (term_cfg.x_min, term_cfg.x_max, term_cfg.y_min, term_cfg.y_max, term_cfg.z_min, term_cfg.z_max)
    for episode in selected:
        saved_path = save_plot(
            episode,
            plot_dir,
            bounds,
            break_at_phase_transition=args.break_at_phase_transition,
            show_phase_reset_jump=args.show_phase_reset_jump,
            showcase_mode=args.showcase_mode,
            plot_sample_rate=args.plot_sample_rate,
            max_annotations=args.max_annotations,
            no_text_annotations=args.no_text_annotations,
            show_callouts=args.show_callouts,
            show_strategy_switch_markers=args.show_strategy_switch_markers,
            show_regime_switch_markers=args.show_regime_switch_markers,
            show_boundary_priority_markers=args.show_boundary_priority_markers,
            show_pursuit_links=args.show_pursuit_links,
            pursuit_link_interval=args.pursuit_link_interval,
            view=args.view,
        )
        option_sequence = "->".join(OPTION_NAMES[index] for index in episode.option_sequence)
        print(
            f"Saved plot: {saved_path} | lowlevel_steps={episode.lowlevel_steps} | "
            f"switch_count={episode.switch_count} | unique_strategies={len(set(episode.option_sequence))} | "
            f"outcome={episode.outcome} | seed={episode.showcase_seed} | option_sequence={option_sequence}"
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
