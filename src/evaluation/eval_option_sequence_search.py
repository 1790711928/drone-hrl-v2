from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from src.training.highlevel_env import COMPOSITE_SCENARIOS, HighLevelOptionEnv


def parse_durations(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    return vals or [8]


def generate_sequences(max_len: int) -> list[tuple[int, ...]]:
    seqs: list[tuple[int, ...]] = []
    for l in range(1, max_len + 1):
        seqs.extend(itertools.product([0, 1, 2, 3], repeat=l))
    return seqs


def run_sequence_episode(env: HighLevelOptionEnv, sequence: tuple[int, ...]) -> tuple[str, float, int]:
    obs, _ = env.reset(options={"scenario_set": "composite"})
    done = False
    total_reward = 0.0
    high_steps = 0
    outcome = "timeout"

    while not done:
        action = sequence[high_steps % len(sequence)]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        high_steps += 1
        done = bool(terminated or truncated)
        outcome = str(info.get("outcome", "timeout"))

    return outcome, total_reward, high_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Option sequence search/oracle baseline on composite scenarios")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--scenario-set", choices=["composite"], default="composite")
    parser.add_argument("--max-seq-len", type=int, default=3)
    parser.add_argument("--option-durations", default="4,6,8,10")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium") from exc

    ckpt_dir = Path(args.checkpoint_dir)
    low_paths = [
        ckpt_dir / "sac_low_1_rear_close_threat.zip",
        ckpt_dir / "sac_low_2_flank_threat.zip",
        ckpt_dir / "sac_low_3_boundary_constrained.zip",
        ckpt_dir / "sac_low_4_vertical_z_threat.zip",
    ]
    for p in low_paths:
        if not p.exists():
            print(f"Missing checkpoint: {p}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    low_models = [SAC.load(str(p)) for p in low_paths]
    durations = parse_durations(args.option_durations)
    sequences = generate_sequences(args.max_seq_len)
    scenario_names = list(COMPOSITE_SCENARIOS.keys())

    rows: list[dict[str, str | float | int]] = []
    best_by_scenario: dict[str, dict[str, str | float | int]] = {}

    for scenario_name in scenario_names:
        for duration in durations:
            env = HighLevelOptionEnv(
                low_models=low_models,
                option_duration=duration,
                switch_penalty=0.0,
                max_highlevel_steps=80,
                scenario_set="composite",
            )

            for seq in sequences:
                succ = cap = oob = 0
                rewards = 0.0
                steps = 0.0

                for _ in range(args.episodes):
                    obs, _ = env.reset(options={"scenario_set": "composite"})
                    # force target composite scenario
                    env.current_scenario_name = scenario_name
                    _ = env._reset_composite_state(scenario_name)

                    done = False
                    ep_reward = 0.0
                    ep_steps = 0
                    outcome = "timeout"
                    while not done:
                        action = seq[ep_steps % len(seq)]
                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_reward += float(reward)
                        ep_steps += 1
                        done = bool(terminated or truncated)
                        outcome = str(info.get("outcome", "timeout"))

                    rewards += ep_reward
                    steps += ep_steps
                    if outcome == "escaped":
                        succ += 1
                    elif outcome == "captured":
                        cap += 1
                    elif outcome == "out_of_bounds":
                        oob += 1

                n = max(args.episodes, 1)
                row = {
                    "scenario": scenario_name,
                    "sequence": "->".join(f"pi{i+1}" for i in seq),
                    "seq_len": len(seq),
                    "option_duration": duration,
                    "success_rate": succ / n,
                    "out_of_bounds_rate": oob / n,
                    "capture_rate": cap / n,
                    "avg_reward": rewards / n,
                    "avg_steps": steps / n,
                }
                rows.append(row)

                best = best_by_scenario.get(scenario_name)
                if best is None or float(row["success_rate"]) > float(best["best_success_rate"]):
                    best_by_scenario[scenario_name] = {
                        "best_sequence": row["sequence"],
                        "best_success_rate": row["success_rate"],
                        "best_option_duration": duration,
                        "out_of_bounds_rate": row["out_of_bounds_rate"],
                        "avg_steps": row["avg_steps"],
                    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "option_sequence_search.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] saved: {out_csv}")
    print("=== Option Sequence Search Summary ===")

    scenario_best_table: dict[str, dict[str, str | float | int]] = {}
    best_rates: list[float] = []
    for scen, best in best_by_scenario.items():
        sr = float(best["best_success_rate"])
        best_rates.append(sr)
        scenario_best_table[scen] = {
            "best_sequence": str(best["best_sequence"]),
            "best_success_rate": sr,
            "best_option_duration": int(best["best_option_duration"]),
            "out_of_bounds_rate": float(best["out_of_bounds_rate"]),
            "avg_steps": float(best["avg_steps"]),
        }
        print(
            f"{scen}: best_sequence={best['best_sequence']}, best_success_rate={sr:.3f}, "
            f"best_option_duration={best['best_option_duration']}, oob={float(best['out_of_bounds_rate']):.3f}, "
            f"avg_steps={float(best['avg_steps']):.2f}"
        )

    mean_best = sum(best_rates) / max(len(best_rates), 1)
    min_best = min(best_rates) if best_rates else 0.0
    all_scenarios_above_fixed = all(r > 0.767 for r in best_rates) if best_rates else False
    mean_best_better_than_fixed = mean_best > 0.767

    print(f"mean_best_success_rate_across_scenarios={mean_best:.3f}")
    print(f"min_best_success_rate_across_scenarios={min_best:.3f}")
    print(f"scenario_best_table={scenario_best_table}")
    print(f"all_scenarios_above_fixed_pi3_baseline={all_scenarios_above_fixed}")
    print(f"mean_best_better_than_fixed_pi3_0.767={mean_best_better_than_fixed}")

    focus_names = ["composite_flank_boundary", "composite_rear_flank_boundary"]
    for fname in focus_names:
        focus_rows = [r for r in rows if str(r["scenario"]) == fname]
        focus_rows.sort(key=lambda r: (float(r["success_rate"]), -float(r["out_of_bounds_rate"]), -float(r["avg_steps"])), reverse=True)
        print(f"top5_{fname}=")
        for fr in focus_rows[:5]:
            print(
                f"  seq={fr['sequence']}, dur={fr['option_duration']}, sr={float(fr['success_rate']):.3f}, "
                f"oob={float(fr['out_of_bounds_rate']):.3f}, steps={float(fr['avg_steps']):.2f}"
            )


if __name__ == "__main__":
    main()
