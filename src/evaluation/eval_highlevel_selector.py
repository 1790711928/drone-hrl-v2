from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.training.highlevel_env import HighLevelOptionEnv


def heuristic_option(obs, prev_option: int | None, hold_steps: int) -> int:
    threat_forward = float(obs[22])
    threat_right = float(obs[23])
    threat_up = float(obs[24])
    min_boundary_margin = float(obs[17])
    distance = float(obs[3])

    boundary_threshold = 0.18
    flank_threshold = 0.50
    vertical_threshold = 0.45
    rear_forward_threshold = -0.60
    rear_close_distance = 0.08

    if min_boundary_margin < boundary_threshold:
        candidate = 2
    elif abs(threat_right) > flank_threshold:
        candidate = 1
    elif abs(threat_up) > vertical_threshold:
        candidate = 3
    elif threat_forward < rear_forward_threshold and distance < rear_close_distance:
        candidate = 0
    else:
        candidate = prev_option if prev_option is not None else 0

    if prev_option is not None and hold_steps < 1:
        return prev_option
    return int(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed/random/high-level selector on scenario sets")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mode", choices=["fixed", "random", "heuristic", "highlevel"], default="fixed")
    parser.add_argument("--fixed-policy", type=int, default=0)
    parser.add_argument("--high-model", default="outputs/checkpoints/ppo_highlevel_switch.zip")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--scenario-set", choices=["basic", "mixed", "composite"], default="composite")
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--max-highlevel-steps", type=int, default=80)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
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
    env = HighLevelOptionEnv(
        low_models=low_models,
        option_duration=args.option_duration,
        switch_penalty=args.switch_penalty,
        max_highlevel_steps=args.max_highlevel_steps,
        scenario_set=args.scenario_set,
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

    for _ in range(args.episodes):
        obs, info = env.reset(options={"scenario_set": args.scenario_set})
        scen = str(info.get("scenario_name", "unknown"))
        scenario_outcomes.setdefault(scen, {"escaped": 0, "captured": 0, "out_of_bounds": 0, "timeout": 0})

        done = False
        ep_reward = 0.0
        ep_steps = 0
        outcome = "timeout"
        prev_option = None
        hold_steps = 0

        while not done:
            if args.mode == "fixed":
                action = int(max(0, min(3, args.fixed_policy)))
            elif args.mode == "random":
                action = random.randint(0, 3)
            elif args.mode == "heuristic":
                action = heuristic_option(obs, prev_option, hold_steps)
            else:
                assert high_model is not None
                action, _ = high_model.predict(obs, deterministic=True)
                action = int(action)

            option_usage[action] += 1
            if prev_option is None or action != prev_option:
                hold_steps = 0
            else:
                hold_steps += 1
            prev_option = action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = bool(terminated or truncated)
            outcome = str(info.get("outcome", "timeout"))

        total_reward += ep_reward
        total_steps += ep_steps
        total_switch += float(info.get("switch_count", 0))

        if outcome == "escaped":
            succ += 1
        elif outcome == "captured":
            cap += 1
        elif outcome == "out_of_bounds":
            oob += 1
        scenario_outcomes[scen][outcome] = scenario_outcomes[scen].get(outcome, 0) + 1

    n = max(args.episodes, 1)
    usage_total = max(sum(option_usage), 1)
    usage_rate = {f"pi{i+1}": option_usage[i] / usage_total for i in range(4)}

    print("=== High-level selector evaluation ===")
    print(f"mode={args.mode}, episodes={args.episodes}, scenario_set={args.scenario_set}")
    print(f"success_rate={succ / n:.3f}")
    print(f"capture_rate={cap / n:.3f}")
    print(f"out_of_bounds_rate={oob / n:.3f}")
    print(f"avg_reward={total_reward / n:.3f}")
    print(f"avg_steps={total_steps / n:.3f}")
    print(f"avg_switch_count={total_switch / n:.3f}")
    print(f"option_usage_rate={usage_rate}")
    print(f"outcome_by_scenario={scenario_outcomes}")


if __name__ == "__main__":
    main()
