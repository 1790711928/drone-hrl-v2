from __future__ import annotations

import argparse

from src.env.pursuit_escape_env import PursuitEscapeEnv
from src.visualization.trajectory import save_trajectory_plot


def run_demo(steps: int, scenario: str, save_plot: bool, plot_path: str) -> None:
    env = PursuitEscapeEnv()
    obs = env.reset(scenario=scenario)
    print(f"[reset] scenario={scenario} distance={obs['distance']:.2f}")

    if env.state is None:
        raise RuntimeError("env state is None after reset")
    evader_points = [(env.state.evader.x, env.state.evader.y, env.state.evader.z)]
    pursuer_points = [(env.state.pursuer.x, env.state.pursuer.y, env.state.pursuer.z)]

    for i in range(steps):
        # 当前是“规则演示”，不是训练：逃跑方动作固定，追击方规则引导。
        obs, reward, done, info = env.step((0.2, 0.05, 0.01))
        if env.state is None:
            raise RuntimeError("env state is None during rollout")
        evader_points.append((env.state.evader.x, env.state.evader.y, env.state.evader.z))
        pursuer_points.append((env.state.pursuer.x, env.state.pursuer.y, env.state.pursuer.z))

        print(
            f"[step {i+1:03d}] distance={info['distance']:.2f} closing={info['closing_speed']:.2f} "
            f"reward={reward:.3f} outcome={info['outcome']} escape_streak={info['escape_streak']}"
        )
        if done:
            break

    if save_plot:
        try:
            saved = save_trajectory_plot(evader_points, pursuer_points, plot_path)
            print(f"[plot] saved trajectory to: {saved}")
        except RuntimeError as exc:
            print(f"[plot] skipped: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D drone pursuit-escape MVP demo")
    parser.add_argument("--scenario", default="rear_close_threat", help="scenario name")
    parser.add_argument("--steps", type=int, default=30, help="max demo steps")
    parser.add_argument("--save-plot", action="store_true", help="save 3D trajectory plot (png)")
    parser.add_argument("--plot-path", default="outputs/trajectory.png", help="output path for trajectory plot")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_demo(steps=args.steps, scenario=args.scenario, save_plot=args.save_plot, plot_path=args.plot_path)
