from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    success: bool
    time_to_escape: int
    min_distance_margin: float
    mean_distance_growth_rate: float
    control_energy: float
    boundary_violation: bool
