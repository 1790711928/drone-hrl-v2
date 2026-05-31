from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.env.dynamics import Agent3DState


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    evader: Agent3DState
    pursuer: Agent3DState


SCENARIOS: Dict[str, ScenarioSpec] = {
    # 1) 后方近距离威胁
    "rear_close_threat": ScenarioSpec(
        name="rear_close_threat",
        evader=Agent3DState(x=0.0, y=0.0, z=12.0, speed=10.0, yaw=0.2, pitch=0.0),
        pursuer=Agent3DState(x=-6.0, y=0.0, z=12.0, speed=11.5, yaw=0.0, pitch=0.0),
    ),
    # 2) 侧翼威胁（单追击机侧后方高威胁）
    "flank_threat": ScenarioSpec(
        name="flank_threat",
        evader=Agent3DState(x=0.0, y=0.0, z=10.0, speed=9.5, yaw=0.0, pitch=0.0),
        pursuer=Agent3DState(x=-8.0, y=8.0, z=10.0, speed=11.0, yaw=-0.3, pitch=0.0),
    ),
    # 3) 边界受限
    "boundary_constrained": ScenarioSpec(
        name="boundary_constrained",
        evader=Agent3DState(x=46.0, y=0.0, z=8.0, speed=9.0, yaw=0.8, pitch=0.0),
        pursuer=Agent3DState(x=30.0, y=-10.0, z=8.0, speed=10.5, yaw=0.5, pitch=0.0),
    ),
    # 4) 垂直 z 轴威胁
    "vertical_z_threat": ScenarioSpec(
        name="vertical_z_threat",
        evader=Agent3DState(x=0.0, y=0.0, z=18.0, speed=10.0, yaw=0.0, pitch=0.16),
        pursuer=Agent3DState(x=-3.0, y=1.0, z=3.0, speed=11.0, yaw=0.05, pitch=0.18),
    ),
}

# backward-compatible aliases
SCENARIOS["s1_close_threat"] = SCENARIOS["rear_close_threat"]
SCENARIOS["s2_energy_management"] = SCENARIOS["flank_threat"]
SCENARIOS["s3_vertical_escape"] = SCENARIOS["vertical_z_threat"]
SCENARIOS["s4_boundary_disturbance"] = SCENARIOS["boundary_constrained"]
SCENARIOS["flank_encirclement"] = SCENARIOS["flank_threat"]
