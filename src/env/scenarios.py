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
    "s1_close_threat": ScenarioSpec(
        name="s1_close_threat",
        evader=Agent3DState(x=0.0, y=0.0, z=10.0, speed=10.0, yaw=0.2, pitch=0.0),
        pursuer=Agent3DState(x=-8.0, y=0.0, z=10.0, speed=11.0, yaw=0.0, pitch=0.0),
    ),
    "s2_energy_management": ScenarioSpec(
        name="s2_energy_management",
        evader=Agent3DState(x=0.0, y=0.0, z=12.0, speed=9.0, yaw=0.0, pitch=0.0),
        pursuer=Agent3DState(x=-20.0, y=-5.0, z=12.0, speed=10.0, yaw=0.1, pitch=0.0),
    ),
    "s3_vertical_escape": ScenarioSpec(
        name="s3_vertical_escape",
        evader=Agent3DState(x=0.0, y=0.0, z=8.0, speed=10.0, yaw=0.0, pitch=0.15),
        pursuer=Agent3DState(x=-15.0, y=2.0, z=5.0, speed=11.0, yaw=0.0, pitch=0.0),
    ),
    "s4_boundary_disturbance": ScenarioSpec(
        name="s4_boundary_disturbance",
        evader=Agent3DState(x=85.0, y=0.0, z=6.0, speed=9.5, yaw=0.7, pitch=0.0),
        pursuer=Agent3DState(x=70.0, y=-10.0, z=6.0, speed=10.5, yaw=0.4, pitch=0.0),
    ),
}
