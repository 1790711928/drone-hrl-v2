from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from src.training.highlevel_env import HighLevelOptionEnv


def heuristic_option(
    obs,
    prev_option: int | None,
    hold_steps: int,
    *,
    boundary_danger_threshold: float,
    boundary_controllable_threshold: float,
    flank_threshold: float,
    vertical_threshold: float,
    rear_distance_threshold: float,
) -> int:
    threat_forward = float(obs[22])
    threat_right = abs(float(obs[23]))
    threat_up = abs(float(obs[24]))
    min_boundary_margin = float(obs[17])
    distance = float(obs[3])

    rear_forward_threshold = -0.60

    # A) danger zone: prioritize boundary recovery
    if min_boundary_margin < boundary_danger_threshold:
        candidate = 2
    # B) controllable zone: handoff from pi3 to threat-specific options
    elif min_boundary_margin >= boundary_controllable_threshold:
        if threat_right > flank_threshold:
            candidate = 1
        elif threat_up > vertical_threshold:
            candidate = 3
        elif threat_forward < rear_forward_threshold and distance < rear_distance_threshold:
            candidate = 0
        else:
            candidate = prev_option if prev_option is not None and prev_option != 2 else 0
    # C) transition zone: allow very strong flank to intervene, else continue boundary recovery
    else:
        very_strong_flank = threat_right > (flank_threshold + 0.10)
        if very_strong_flank:
            candidate = 1
        elif threat_up > (vertical_threshold + 0.08):
            candidate = 3
        else:
            candidate = 2

    # hysteresis: keep current option for at least one high-level step unless clearly better
    if prev_option is not None and hold_steps < 1 and candidate != prev_option:
        return prev_option
    return int(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed/random/heuristic/high-level selector on scenario sets")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mode", choices=["fixed", "random", "heuristic", "highlevel"], default="fixed")
    parser.add_argument("--fixed-policy", type=int, default=0)
    parser.add_argument("--high-model", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite", "sequential", "continuous_pursuit"], default="composite")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    parser.add_argument("--boundary-danger-threshold", type=float, default=0.15)
    parser.add_argument("--boundary-controllable-threshold", type=float, default=0.25)
    parser.add_argument("--flank-threshold", type=float, default=0.65)
    parser.add_argument("--vertical-threshold", type=float, default=0.55)
    parser.add_argument("--rear-distance-threshold", type=float, default=0.09)
    parser.add_argument("--episode-lowlevel-steps", type=int, default=400)
    parser.add_argument("--regime-duration", type=int, default=60)
    parser.add_argument("--pursuer-speed-ratio", type=float, default=1.25)
    parser.add_argument("--regime-schedule", default="rear,vertical,boundary,flank,rear,boundary")
    args = parser.parse_args()

    from stable_baselines3 import PPO, SAC

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
    )

    high_model = None
    if args.mode == "highlevel":
        high_path = Path(args.high_model)
        if not high_path.exists():
            print(f"Missing checkpoint: {high_path}")
            print("Please run this script locally after training high-level PPO selector.")
            return
        high_model = PPO.load(str(high_path))

    succ = cap = oob = 0
    total_reward = total_steps = total_switch = 0.0
    option_usage = [0, 0, 0, 0]
    scenario_outcomes: dict[str, dict[str, int]] = {}
    scenario_option_usage: dict[str, list[int]] = {}
    scenario_switch_counts: dict[str, list[float]] = {}
    scenario_first_options: dict[str, list[int]] = {}
    scenario_sequences: dict[str, Counter[str]] = {}
    completed_phases_total = 0.0
    total_phases_total = 0.0
    phase_success_by_type: dict[str, int] = {}
    phase_failure_by_type: dict[str, int] = {}
    regime_option_usage: dict[str, list[int]] = {}
    continuous_lowlevel_steps_total = 0.0
    final_distance_total = 0.0
    recent_distance_total = 0.0
    recent_closing_total = 0.0
    regime_coverage_total = 0.0
    timeout = 0

    for _ in range(args.episodes):
        obs, info = env.reset(options={"scenario_set": args.scenario_set})
        scen = str(info.get("scenario_name", "unknown"))
        scenario_outcomes.setdefault(scen, {"escaped": 0, "captured": 0, "out_of_bounds": 0, "timeout": 0})
        scenario_option_usage.setdefault(scen, [0, 0, 0, 0])
        scenario_switch_counts.setdefault(scen, [])
        scenario_first_options.setdefault(scen, [])
        scenario_sequences.setdefault(scen, Counter())

        done = False
        ep_reward = 0.0
        ep_steps = 0
        outcome = "timeout"
        prev_option = None
        hold_steps = 0
        seq: list[int] = []

        while not done:
            if args.mode == "fixed":
                action = int(max(0, min(3, args.fixed_policy)))
            elif args.mode == "random":
                action = random.randint(0, 3)
            elif args.mode == "heuristic":
                action = heuristic_option(
                    obs,
                    prev_option,
                    hold_steps,
                    boundary_danger_threshold=args.boundary_danger_threshold,
                    boundary_controllable_threshold=args.boundary_controllable_threshold,
                    flank_threshold=args.flank_threshold,
                    vertical_threshold=args.vertical_threshold,
                    rear_distance_threshold=args.rear_distance_threshold,
                )
            else:
                assert high_model is not None
                action, _ = high_model.predict(obs, deterministic=True)
                action = int(action)

            option_usage[action] += 1
            scenario_option_usage[scen][action] += 1
            if not seq:
                scenario_first_options[scen].append(action)

            if prev_option is None or action != prev_option:
                hold_steps = 0
            else:
                hold_steps += 1
            prev_option = action
            seq.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            regime_name = str(info.get("regime_name", "none"))
            if args.scenario_set == "continuous_pursuit":
                regime_steps = dict(info.get("regime_lowlevel_steps", {regime_name: 1}))
                for regime, step_count in regime_steps.items():
                    regime_key = str(regime)
                    regime_option_usage.setdefault(regime_key, [0, 0, 0, 0])
                    regime_option_usage[regime_key][action] += int(step_count)
            ep_reward += float(reward)
            ep_steps += 1
            done = bool(terminated or truncated)
            outcome = str(info.get("outcome", "timeout"))

        total_reward += ep_reward
        total_steps += ep_steps
        sc = float(info.get("switch_count", 0))
        total_switch += sc
        scenario_switch_counts[scen].append(sc)
        completed_phases_total += float(info.get("completed_phases", 0))
        total_phases_total += float(info.get("total_phases", 0))
        for phase, count in dict(info.get("phase_success_by_phase_type", {})).items():
            phase_success_by_type[str(phase)] = phase_success_by_type.get(str(phase), 0) + int(count)
        for phase, count in dict(info.get("phase_failure_by_phase_type", {})).items():
            phase_failure_by_type[str(phase)] = phase_failure_by_type.get(str(phase), 0) + int(count)
        continuous_lowlevel_steps_total += float(info.get("continuous_lowlevel_steps", 0))
        final_distance_total += float(info.get("final_distance", 0.0))
        recent_distance_total += float(info.get("recent_distance", 0.0))
        recent_closing_total += float(info.get("recent_closing_speed", 0.0))
        regime_coverage_total += float(info.get("regime_coverage_rate", 0.0))

        seq_key = "->".join(f"pi{a+1}" for a in seq[:6])
        scenario_sequences[scen][seq_key] += 1

        if outcome == "escaped":
            succ += 1
        elif outcome == "captured":
            cap += 1
        elif outcome == "out_of_bounds":
            oob += 1
        else:
            timeout += 1
        scenario_outcomes[scen][outcome] = scenario_outcomes[scen].get(outcome, 0) + 1

    n = max(args.episodes, 1)
    usage_total = max(sum(option_usage), 1)
    usage_rate = {f"pi{i+1}": option_usage[i] / usage_total for i in range(4)}

    usage_by_scenario: dict[str, dict[str, float]] = {}
    avg_switch_by_scenario: dict[str, float] = {}
    first_option_by_scenario: dict[str, dict[str, float]] = {}
    common_seq_by_scenario: dict[str, list[tuple[str, int]]] = {}

    for scen, counts in scenario_option_usage.items():
        total = max(sum(counts), 1)
        usage_by_scenario[scen] = {f"pi{i+1}": counts[i] / total for i in range(4)}
        sw = scenario_switch_counts.get(scen, [])
        avg_switch_by_scenario[scen] = sum(sw) / max(len(sw), 1)

        first_counts = [0, 0, 0, 0]
        for idx in scenario_first_options.get(scen, []):
            first_counts[idx] += 1
        first_total = max(sum(first_counts), 1)
        first_option_by_scenario[scen] = {f"pi{i+1}": first_counts[i] / first_total for i in range(4)}

        common_seq_by_scenario[scen] = scenario_sequences[scen].most_common(3)

    print("=== High-level selector evaluation ===")
    print(f"mode={args.mode}, episodes={args.episodes}, scenario_set={args.scenario_set}")
    print(f"success_rate={succ / n:.3f}")
    print(f"capture_rate={cap / n:.3f}")
    print(f"out_of_bounds_rate={oob / n:.3f}")
    print(f"avg_reward={total_reward / n:.3f}")
    print(f"avg_steps={total_steps / n:.3f}")
    print(f"avg_switch_count={total_switch / n:.3f}")
    print(f"timeout_rate={timeout / n:.3f}")
    print(f"option_usage_rate={usage_rate}")
    if args.scenario_set == "continuous_pursuit":
        usage_by_regime = {}
        for regime, counts in regime_option_usage.items():
            total = max(sum(counts), 1)
            usage_by_regime[regime] = {f"pi{i+1}": counts[i] / total for i in range(4)}
        print(f"option_usage_by_regime={usage_by_regime}")
        print(f"avg_episode_lowlevel_steps={continuous_lowlevel_steps_total / n:.3f}")
        print(f"avg_final_distance={final_distance_total / n:.3f}")
        print(f"avg_recent_distance={recent_distance_total / n:.3f}")
        print(f"avg_recent_closing_speed={recent_closing_total / n:.3f}")
        print(f"regime_coverage_rate={regime_coverage_total / n:.3f}")
    print(f"outcome_by_scenario={scenario_outcomes}")
    print(f"option_usage_by_scenario={usage_by_scenario}")
    print(f"avg_switch_count_by_scenario={avg_switch_by_scenario}")
    print(f"first_option_by_scenario={first_option_by_scenario}")
    print(f"common_option_sequences_by_scenario={common_seq_by_scenario}")
    print(f"phase_completion_rate={completed_phases_total / max(total_phases_total, 1.0):.3f}")
    print(f"avg_completed_phases={completed_phases_total / n:.3f}")
    print(f"phase_success_by_phase_type={phase_success_by_type}")
    print(f"phase_failure_by_phase_type={phase_failure_by_type}")


if __name__ == "__main__":
    main()
