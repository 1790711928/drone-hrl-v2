from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from src.training.highlevel_env import COMPOSITE_SCENARIOS, SEQUENTIAL_SCENARIOS, HighLevelOptionEnv


def parse_durations(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    return [value for value in values if value > 0] or [8]


def generate_sequences(max_len: int) -> list[tuple[int, ...]]:
    sequences: list[tuple[int, ...]] = []
    for length in range(1, max_len + 1):
        sequences.extend(itertools.product([0, 1, 2, 3], repeat=length))
    return sequences


def row_rank(row: dict[str, str | float | int]) -> tuple[float, float, float, float]:
    return (
        float(row["success_rate"]),
        float(row["avg_completed_phases"]),
        -float(row["out_of_bounds_rate"]),
        -float(row["avg_steps"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Option sequence search/oracle baseline")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--scenario-set", choices=["composite", "sequential"], default="composite")
    parser.add_argument("--max-seq-len", type=int, default=3)
    parser.add_argument("--option-durations", default="4,6,8,10")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    checkpoint_dir = Path(args.checkpoint_dir)
    low_paths = [
        checkpoint_dir / "sac_low_1_rear_close_threat.zip",
        checkpoint_dir / "sac_low_2_flank_threat.zip",
        checkpoint_dir / "sac_low_3_boundary_constrained.zip",
        checkpoint_dir / "sac_low_4_vertical_z_threat.zip",
    ]
    for path in low_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    low_models = [SAC.load(str(path)) for path in low_paths]
    durations = parse_durations(args.option_durations)
    sequences = generate_sequences(args.max_seq_len)
    scenario_names = list(COMPOSITE_SCENARIOS if args.scenario_set == "composite" else SEQUENTIAL_SCENARIOS)

    rows: list[dict[str, str | float | int]] = []
    best_by_scenario: dict[str, dict[str, str | float | int]] = {}

    for scenario_name in scenario_names:
        for duration in durations:
            env = HighLevelOptionEnv(
                low_models=low_models,
                option_duration=duration,
                switch_penalty=0.0,
                max_highlevel_steps=80,
                scenario_set=args.scenario_set,
            )
            for sequence in sequences:
                success = captures = out_of_bounds = 0
                rewards = steps = completed_phases = 0.0
                total_phases = 0.0

                for _ in range(args.episodes):
                    obs, _ = env.reset(options={"scenario_set": args.scenario_set, "scenario_name": scenario_name})
                    done = False
                    ep_reward = 0.0
                    ep_steps = 0
                    info: dict[str, object] = {"outcome": "timeout"}
                    while not done:
                        action = sequence[ep_steps % len(sequence)]
                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_reward += float(reward)
                        ep_steps += 1
                        done = bool(terminated or truncated)

                    outcome = str(info.get("outcome", "timeout"))
                    rewards += ep_reward
                    steps += ep_steps
                    completed_phases += float(info.get("completed_phases", 0))
                    total_phases += float(info.get("total_phases", 0))
                    if outcome == "escaped":
                        success += 1
                    elif outcome == "captured":
                        captures += 1
                    elif outcome == "out_of_bounds":
                        out_of_bounds += 1

                episodes = max(args.episodes, 1)
                row: dict[str, str | float | int] = {
                    "scenario_set": args.scenario_set,
                    "scenario": scenario_name,
                    "sequence": "->".join(f"pi{option + 1}" for option in sequence),
                    "seq_len": len(sequence),
                    "option_duration": duration,
                    "success_rate": success / episodes,
                    "out_of_bounds_rate": out_of_bounds / episodes,
                    "capture_rate": captures / episodes,
                    "avg_reward": rewards / episodes,
                    "avg_steps": steps / episodes,
                    "phase_completion_rate": completed_phases / max(total_phases, 1.0),
                    "avg_completed_phases": completed_phases / episodes,
                }
                rows.append(row)
                current_best = best_by_scenario.get(scenario_name)
                if current_best is None or row_rank(row) > row_rank(current_best):
                    best_by_scenario[scenario_name] = row

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "option_sequence_search.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] saved: {out_csv}")
    print("=== Option Sequence Search Summary ===")
    scenario_best_table: dict[str, dict[str, str | float | int]] = {}
    best_rates: list[float] = []
    for scenario, best in best_by_scenario.items():
        rate = float(best["success_rate"])
        best_rates.append(rate)
        scenario_best_table[scenario] = {
            "best_sequence": str(best["sequence"]),
            "best_success_rate": rate,
            "best_option_duration": int(best["option_duration"]),
            "out_of_bounds_rate": float(best["out_of_bounds_rate"]),
            "avg_steps": float(best["avg_steps"]),
            "phase_completion_rate": float(best["phase_completion_rate"]),
        }
        print(f"{scenario}: {scenario_best_table[scenario]}")

    mean_best = sum(best_rates) / max(len(best_rates), 1)
    min_best = min(best_rates) if best_rates else 0.0
    print(f"mean_best_success_rate_across_scenarios={mean_best:.3f}")
    print(f"min_best_success_rate_across_scenarios={min_best:.3f}")
    print(f"scenario_best_table={scenario_best_table}")
    print(f"all_scenarios_above_fixed_pi3_baseline={bool(best_rates) and all(rate > 0.767 for rate in best_rates)}")
    print(f"mean_best_better_than_fixed_pi3_0.767={mean_best > 0.767}")

    focus_names = (
        ["composite_flank_boundary", "composite_rear_flank_boundary"]
        if args.scenario_set == "composite"
        else list(SEQUENTIAL_SCENARIOS.keys())
    )
    for focus_name in focus_names:
        focus_rows = [row for row in rows if row["scenario"] == focus_name]
        focus_rows.sort(key=row_rank, reverse=True)
        print(f"top5_{focus_name}=")
        for row in focus_rows[:5]:
            print(
                f"  seq={row['sequence']}, dur={row['option_duration']}, sr={float(row['success_rate']):.3f}, "
                f"oob={float(row['out_of_bounds_rate']):.3f}, steps={float(row['avg_steps']):.2f}, "
                f"phase_rate={float(row['phase_completion_rate']):.3f}"
            )


if __name__ == "__main__":
    main()
