from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


def save_trajectory_plot(
    evader_points: Sequence[Tuple[float, float, float]],
    pursuer_points: Sequence[Tuple[float, float, float]],
    output_path: str,
) -> str:
    """Save a 3D trajectory plot as a PNG file."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "matplotlib is required for trajectory plotting. Install with: pip install matplotlib"
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ex, ey, ez = zip(*evader_points)
    px, py, pz = zip(*pursuer_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ex, ey, ez, label="Evader", linewidth=2)
    ax.plot(px, py, pz, label="Pursuer", linewidth=2)

    ax.scatter(ex[0], ey[0], ez[0], marker="o", s=40, label="Evader Start")
    ax.scatter(px[0], py[0], pz[0], marker="o", s=40, label="Pursuer Start")
    ax.scatter(ex[-1], ey[-1], ez[-1], marker="^", s=50, label="Evader End")
    ax.scatter(px[-1], py[-1], pz[-1], marker="^", s=50, label="Pursuer End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Pursuit-Escape Trajectory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
