from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import Any

from src.evaluation.plot_highlevel_trajectories import (
    EpisodePlotData,
    OPTION_NAMES,
    SCRIPTED_SHOWCASE_SCRIPT_NAME,
    TrajectoryRecorder,
    rollout_episode,
    save_plot,
)
from src.training.highlevel_env import HighLevelOptionEnv, SCRIPTED_SHOWCASE_SCENARIO

LOW_MODEL_FILENAMES = (
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
)
SUMMARY_FIELDS = (
    "rank",
    "episode_id",
    "outcome",
    "lowlevel_steps",
    "switch_count",
    "unique_strategies",
    "option_sequence",
    "actual_regime_sequence",
    "showcase_script",
    "seed",
    "showcase_seed",
    "avg_pursuer_evader_distance",
    "max_pursuer_evader_distance",
    "tail_avg_pursuer_evader_distance",
    "x_range",
    "y_range",
    "z_range",
    "axis_balance_score",
    "min_strategy_segment_length",
    "mean_strategy_segment_length",
    "selection_score",
    "selected_rank",
    "selected_flag",
    "plot_3d_path",
    "plot_top_path",
)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _axis_ranges(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0
    xs, ys, zs = zip(*points)
    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)


def _axis_balance_score(x_range: float, y_range: float, z_range: float) -> float:
    ranges = [value for value in (x_range, y_range, z_range) if value > 1e-6]
    if len(ranges) < 2:
        return 0.0
    return max(0.0, min(1.0, min(ranges) / max(ranges)))


def _compressed_names(names: list[str]) -> list[str]:
    compressed: list[str] = []
    for name in names:
        if not compressed or compressed[-1] != name:
            compressed.append(name)
    return compressed


def _option_segments(option_switches: list[tuple[int, str]], point_count: int) -> list[tuple[int, int, str]]:
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


def _strategy_segment_lengths(episode: EpisodePlotData) -> list[float]:
    lengths = [float(max(0, end - start)) for start, end, _ in _option_segments(episode.option_switches, len(episode.evader_points))]
    return lengths or [0.0]


def score_episode(episode: EpisodePlotData) -> dict[str, Any]:
    pair_count = min(len(episode.evader_points), len(episode.pursuer_points))
    distances = [_distance(episode.evader_points[index], episode.pursuer_points[index]) for index in range(pair_count)]
    avg_distance = sum(distances) / len(distances) if distances else 0.0
    max_distance = max(distances) if distances else 0.0
    tail_start = int(0.8 * len(distances)) if distances else 0
    tail = distances[tail_start:] or distances
    tail_avg_distance = sum(tail) / len(tail) if tail else 0.0

    x_range, y_range, z_range = _axis_ranges(episode.evader_points)
    axis_balance = _axis_balance_score(x_range, y_range, z_range)
    segment_lengths = _strategy_segment_lengths(episode)
    min_segment_length = min(segment_lengths)
    mean_segment_length = sum(segment_lengths) / len(segment_lengths)
    unique_strategies = len(set(episode.option_sequence))

    steps_score = min(float(episode.lowlevel_steps), 400.0) * 0.25
    strategy_score = 40.0 * unique_strategies + 8.0 * min(float(episode.switch_count), 6.0)
    outcome_score = {
        "escaped": 25.0,
        "timeout": 10.0,
        "out_of_bounds": -20.0,
        "captured": -30.0,
    }.get(episode.outcome, 0.0)
    distance_score = -0.35 * avg_distance - 0.25 * tail_avg_distance - 0.10 * max_distance
    axis_balance_bonus = 25.0 * axis_balance
    segment_score = 0.20 * mean_segment_length + 0.30 * min_segment_length
    penalty = 0.0
    if unique_strategies < 4:
        penalty += 80.0
    if episode.switch_count < 3:
        penalty += 40.0
    penalty += max(0.0, tail_avg_distance - 55.0) * 1.5
    penalty += max(0.0, max_distance - 90.0) * 1.0
    selection_score = (
        steps_score
        + strategy_score
        + outcome_score
        + distance_score
        + axis_balance_bonus
        + segment_score
        - penalty
    )

    option_sequence = "->".join(OPTION_NAMES[index] for index in episode.option_sequence)
    actual_regime_sequence = "->".join(_compressed_names([name for _, name in episode.regime_starts]))
    return {
        "rank": "",
        "episode_id": episode.episode_id,
        "outcome": episode.outcome,
        "lowlevel_steps": episode.lowlevel_steps,
        "switch_count": episode.switch_count,
        "unique_strategies": unique_strategies,
        "option_sequence": option_sequence,
        "actual_regime_sequence": actual_regime_sequence,
        "showcase_script": SCRIPTED_SHOWCASE_SCRIPT_NAME,
        "seed": episode.showcase_seed,
        "showcase_seed": episode.showcase_seed,
        "avg_pursuer_evader_distance": avg_distance,
        "max_pursuer_evader_distance": max_distance,
        "tail_avg_pursuer_evader_distance": tail_avg_distance,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "axis_balance_score": axis_balance,
        "min_strategy_segment_length": min_segment_length,
        "mean_strategy_segment_length": mean_segment_length,
        "selection_score": selection_score,
        "selected_rank": "",
        "selected_flag": 0,
        "plot_3d_path": "",
        "plot_top_path": "",
    }


def _format_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted = dict(row)
    for key in (
        "avg_pursuer_evader_distance",
        "max_pursuer_evader_distance",
        "tail_avg_pursuer_evader_distance",
        "x_range",
        "y_range",
        "z_range",
        "axis_balance_score",
        "min_strategy_segment_length",
        "mean_strategy_segment_length",
        "selection_score",
    ):
        formatted[key] = f"{float(formatted[key]):.3f}"
    return formatted


def _print_top_table(rows: list[dict[str, Any]], limit: int = 10) -> None:
    print("=== Top scripted_showcase candidates ===")
    print(
        "rank, selection_score, lowlevel_steps, switch_count, unique_strategies, "
        "outcome, avg_distance, tail_avg_distance, max_distance, axis_balance_score, seed"
    )
    for row in rows[:limit]:
        print(
            f"{row['rank']}, {row['selection_score']:.3f}, {row['lowlevel_steps']}, {row['switch_count']}, "
            f"{row['unique_strategies']}, {row['outcome']}, "
            f"{row['avg_pursuer_evader_distance']:.3f}, "
            f"{row['tail_avg_pursuer_evader_distance']:.3f}, "
            f"{row['max_pursuer_evader_distance']:.3f}, "
            f"{row['axis_balance_score']:.3f}, {row['showcase_seed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select scripted_showcase trajectory candidates by presentation quality")
    parser.add_argument("--attempts", type=int, default=120)
    parser.add_argument("--max-plots", type=int, default=3)
    parser.add_argument("--episode-lowlevel-steps", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation/scripted_showcase_selection")
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.35)
    parser.add_argument("--showcase-bound-scale", type=float, default=6.0)
    parser.add_argument("--showcase-z-bound-scale", type=float, default=6.0)
    parser.add_argument("--plot-sample-rate", type=int, default=5)
    parser.add_argument("--view", choices=["3d", "topdown"], default="3d")
    parser.add_argument(
        "--save-top-view",
        action="store_true",
        help="Also save a top-view x-y plot for each selected scripted_showcase candidate.",
    )
    args = parser.parse_args()

    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.max_plots <= 0:
        parser.error("--max-plots must be positive")
    if args.episode_lowlevel_steps <= 0:
        parser.error("--episode-lowlevel-steps must be positive")
    if args.plot_sample_rate <= 0:
        parser.error("--plot-sample-rate must be positive")
    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib is not installed. Please install matplotlib to save scripted_showcase candidate plots.")
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    low_paths = [checkpoint_dir / filename for filename in LOW_MODEL_FILENAMES]
    for path in low_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    from stable_baselines3 import SAC

    low_models = [SAC.load(str(path)) for path in low_paths]
    env = HighLevelOptionEnv(
        low_models=low_models,
        option_duration=8,
        max_highlevel_steps=max(80, args.episode_lowlevel_steps // 8 + 5),
        scenario_set=SCRIPTED_SHOWCASE_SCENARIO,
        episode_lowlevel_steps=args.episode_lowlevel_steps,
        pursuer_speed_ratio=args.pursuer_speed_ratio,
        showcase_bound_scale=args.showcase_bound_scale,
        showcase_z_bound_scale=args.showcase_z_bound_scale,
    )
    recorder = TrajectoryRecorder(env)
    recorder.attach()

    episodes: list[EpisodePlotData] = []
    rows: list[dict[str, Any]] = []
    for episode_id in range(1, args.attempts + 1):
        episode = rollout_episode(
            env,
            recorder,
            episode_id=episode_id,
            mode="scripted_showcase",
            fixed_policy=0,
            high_model=None,
            scenario_set=SCRIPTED_SHOWCASE_SCENARIO,
        )
        episodes.append(episode)
        rows.append(score_episode(episode))

    ranked_pairs = sorted(zip(rows, episodes), key=lambda pair: pair[0]["selection_score"], reverse=True)
    for rank, (row, _) in enumerate(ranked_pairs, start=1):
        row["rank"] = rank
    selected_pairs = ranked_pairs[: args.max_plots]
    for selected_rank, (row, _) in enumerate(selected_pairs, start=1):
        row["selected_rank"] = selected_rank
        row["selected_flag"] = 1

    out_dir = Path(args.out_dir)
    if args.save_top_view:
        plots_3d_dir = out_dir / "plots_3d"
        plots_top_dir = out_dir / "plots_top"
    else:
        plots_3d_dir = out_dir / "plots"
        plots_top_dir = out_dir / "plots"

    for row, episode in selected_pairs:
        if args.save_top_view:
            saved_3d_path = save_plot(
                episode,
                plots_3d_dir,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                plot_sample_rate=args.plot_sample_rate,
                view="3d",
            )
            saved_top_path = save_plot(
                episode,
                plots_top_dir,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                plot_sample_rate=args.plot_sample_rate,
                view="topdown",
            )
            row["plot_3d_path"] = str(saved_3d_path)
            row["plot_top_path"] = str(saved_top_path)
            saved_path_text = f"3d={saved_3d_path} | top={saved_top_path}"
        else:
            saved_path = save_plot(
                episode,
                plots_3d_dir,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                plot_sample_rate=args.plot_sample_rate,
                view=args.view,
            )
            if args.view == "topdown":
                row["plot_top_path"] = str(saved_path)
            else:
                row["plot_3d_path"] = str(saved_path)
            saved_path_text = str(saved_path)
        print(
            f"Saved selected plot rank={row['selected_rank']}: {saved_path_text} | "
            f"score={row['selection_score']:.3f} | lowlevel_steps={row['lowlevel_steps']} | "
            f"switch_count={row['switch_count']} | unique_strategies={row['unique_strategies']} | "
            f"outcome={row['outcome']} | seed={row['showcase_seed']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scripted_showcase_candidate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(_format_row(row) for row in sorted(rows, key=lambda item: int(item["rank"])))
    print(f"Saved summary CSV: {csv_path}")
    _print_top_table([row for row, _ in ranked_pairs], limit=10)


if __name__ == "__main__":
    main()
