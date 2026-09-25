from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


SCENARIOS = [
    "rear_close_threat",
    "flank_threat",
    "boundary_constrained",
    "vertical_z_threat",
]

MODEL_FILENAMES = [
    "sac_low_1_rear_close_threat.zip",
    "sac_low_2_flank_threat.zip",
    "sac_low_3_boundary_constrained.zip",
    "sac_low_4_vertical_z_threat.zip",
]


@dataclass
class EvalResult:
    success_rate: float
    avg_reward: float
    avg_steps: float
    capture_rate: float
    timeout_rate: float
    out_of_bounds_rate: float


def evaluate_model_on_scenario(model, scenario: str, episodes: int) -> EvalResult:
    from src.training.sac_env import PursuitEscapeGymEnv

    successes = 0
    captures = 0
    timeouts = 0
    out_of_bounds = 0
    rewards: list[float] = []
    steps: list[int] = []

    for _ in range(episodes):
        env = PursuitEscapeGymEnv(scenario=scenario)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        info = {"outcome": "timeout"}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        outcome = str(info.get("outcome", "timeout"))
        if outcome == "escaped":
            successes += 1
        elif outcome == "captured":
            captures += 1
        elif outcome == "out_of_bounds":
            out_of_bounds += 1
        elif outcome == "timeout":
            timeouts += 1

        rewards.append(ep_reward)
        steps.append(ep_steps)

    denom = max(episodes, 1)
    return EvalResult(
        success_rate=successes / denom,
        avg_reward=sum(rewards) / max(len(rewards), 1),
        avg_steps=sum(steps) / max(len(steps), 1),
        capture_rate=captures / denom,
        timeout_rate=timeouts / denom,
        out_of_bounds_rate=out_of_bounds / denom,
    )


def save_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "policy",
        "success_rate",
        "avg_reward",
        "avg_steps",
        "capture_rate",
        "timeout_rate",
        "out_of_bounds_rate",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_save_heatmap(success_matrix: list[list[float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[heatmap] matplotlib not installed, skipping heatmap export.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(success_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(4), labels=[f"pi{i}" for i in range(1, 5)])
    ax.set_yticks(range(4), labels=[f"S{i}" for i in range(1, 5)])
    ax.set_xlabel("Policy")
    ax.set_ylabel("Scenario")
    ax.set_title("Low-level SAC Success Matrix")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Success Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[heatmap] saved: {out_path}")


def print_diagonal_dominance(success_matrix: list[list[float]]) -> None:
    print("\n=== Diagonal Dominance Diagnostics ===")
    for i, row in enumerate(success_matrix):
        owner_idx = i
        owner_sr = row[owner_idx]
        sorted_pairs = sorted(enumerate(row), key=lambda x: x[1], reverse=True)
        best_idx, best_sr = sorted_pairs[0]
        second_sr = sorted_pairs[1][1] if len(sorted_pairs) > 1 else sorted_pairs[0][1]
        margin = owner_sr - second_sr

        print(
            f"S{i+1}({SCENARIOS[i]}): owner=pi{i+1} sr={owner_sr:.3f}, "
            f"best=pi{best_idx+1} sr={best_sr:.3f}, owner_minus_second={margin:.3f}"
        )
        if best_idx != owner_idx:
            print(
                f"WARNING: diagonal dominance violated on {SCENARIOS[i]} "
                f"(owner pi{i+1} is not top-1)."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level SAC diagnostics (4x4 matrix + dominance checks)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    checkpoint_dir = Path(args.checkpoint_dir)
    model_paths = [checkpoint_dir / filename for filename in MODEL_FILENAMES]
    for path in model_paths:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            print("Please run this script locally after training low-level SAC policies.")
            return

    models = [SAC.load(str(path)) for path in model_paths]

    print("=== Low-level Diagnostics Matrix ===")
    rows: list[dict[str, str | float]] = []
    success_matrix: list[list[float]] = []

    for i, scenario in enumerate(SCENARIOS, start=1):
        row_results: list[EvalResult] = []
        for j, model in enumerate(models, start=1):
            result = evaluate_model_on_scenario(model, scenario, args.episodes)
            row_results.append(result)
            rows.append(
                {
                    "scenario": scenario,
                    "policy": f"pi{j}",
                    "success_rate": result.success_rate,
                    "avg_reward": result.avg_reward,
                    "avg_steps": result.avg_steps,
                    "capture_rate": result.capture_rate,
                    "timeout_rate": result.timeout_rate,
                    "out_of_bounds_rate": result.out_of_bounds_rate,
                }
            )

        success_matrix.append([r.success_rate for r in row_results])
        row_line = " | ".join(
            [
                (
                    f"pi{k}: sr={r.success_rate:.2f}, R={r.avg_reward:.1f}, steps={r.avg_steps:.1f}, "
                    f"cap={r.capture_rate:.2f}, to={r.timeout_rate:.2f}, oob={r.out_of_bounds_rate:.2f}"
                )
                for k, r in enumerate(row_results, start=1)
            ]
        )
        print(f"S{i}({scenario}) -> {row_line}")

    out_csv = Path("outputs/evaluation/lowlevel_diagnostics.csv")
    save_csv(rows, out_csv)
    print(f"\n[csv] saved: {out_csv}")

    out_img = Path("outputs/evaluation/lowlevel_success_matrix.png")
    maybe_save_heatmap(success_matrix, out_img)

    print_diagonal_dominance(success_matrix)


if __name__ == "__main__":
    main()
