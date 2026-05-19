from __future__ import annotations

from dataclasses import dataclass

from src.env.termination import EpisodeOutcome


@dataclass(frozen=True)
class RewardWeights:
    w_distance: float
    w_energy: float
    w_survive: float
    w_boundary_risk: float


REWARD_PROFILES = {
    "rear_close_threat": RewardWeights(w_distance=1.2, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.01),
    "flank_threat": RewardWeights(w_distance=1.0, w_energy=0.0012, w_survive=0.02, w_boundary_risk=0.03),
    "boundary_constrained": RewardWeights(w_distance=0.9, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.08),
    "vertical_z_threat": RewardWeights(w_distance=1.25, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.04),
    # backward-compatible aliases
    "s1_close_threat": RewardWeights(w_distance=1.2, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.01),
    "s2_energy_management": RewardWeights(w_distance=1.0, w_energy=0.0012, w_survive=0.02, w_boundary_risk=0.03),
    "flank_encirclement": RewardWeights(w_distance=1.0, w_energy=0.0012, w_survive=0.02, w_boundary_risk=0.01),
    "s3_vertical_escape": RewardWeights(w_distance=1.25, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.04),
    "s4_boundary_disturbance": RewardWeights(w_distance=0.9, w_energy=0.0010, w_survive=0.02, w_boundary_risk=0.08),
}


def boundary_risk(x: float, y: float, z: float, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float) -> float:
    """Soft boundary risk in [0, 1], rises early and sharply near boundaries."""
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    dx = min(x - x_min, x_max - x)
    dy = min(y - y_min, y_max - y)
    dz = min(z - z_min, z_max - z)
    min_norm = min(dx / (0.5 * x_span), dy / (0.5 * y_span), dz / z_span)
    min_norm = max(0.0, min(1.0, min_norm))
    safety_threshold = 0.35
    if min_norm >= safety_threshold:
        return 0.0
    ratio = (safety_threshold - min_norm) / safety_threshold
    return float(min(1.0, ratio * ratio))


def compute_reward_terms(
    *,
    scenario: str,
    prev_distance: float,
    cur_distance: float,
    action: tuple[float, float, float],
    evader_position: tuple[float, float, float],
    pursuer_position: tuple[float, float, float],
    bounds: tuple[float, float, float, float, float, float],
    outcome: EpisodeOutcome,
) -> dict[str, float]:
    w = REWARD_PROFILES.get(scenario, REWARD_PROFILES["rear_close_threat"])
    ev_accel, ev_yaw_rate, ev_pitch_rate = action
    x, y, z = evader_position
    _, _, pz = pursuer_position
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    distance_term = w.w_distance * (cur_distance - prev_distance)
    energy_term = -w.w_energy * (ev_accel**2 + ev_yaw_rate**2 + ev_pitch_rate**2)
    survive_term = w.w_survive
    boundary_term = -w.w_boundary_risk * boundary_risk(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max)
    scenario_term = 0.0
    if scenario == "vertical_z_threat":
        z_span = max(z_max - z_min, 1e-6)
        vertical_sep = abs(z - pz) / z_span
        vertical_sep = min(vertical_sep, 0.5)
        scenario_term += 0.10 * vertical_sep

        z_margin = min(z - z_min, z_max - z)
        z_margin_norm = z_margin / max(0.5 * z_span, 1e-6)
        edge_over = max(0.0, 0.30 - z_margin_norm)
        scenario_term += -0.25 * (edge_over**2)

    terminal_bonus = 0.0
    if outcome == EpisodeOutcome.ESCAPED:
        terminal_bonus = 50.0
    elif outcome == EpisodeOutcome.CAPTURED:
        terminal_bonus = -50.0
    elif outcome == EpisodeOutcome.OUT_OF_BOUNDS:
        terminal_bonus = -20.0
    total_reward = distance_term + energy_term + survive_term + boundary_term + scenario_term + terminal_bonus
    return {
        "distance_term": float(distance_term),
        "energy_term": float(energy_term),
        "survive_term": float(survive_term),
        "boundary_term": float(boundary_term),
        "scenario_term": float(scenario_term),
        "terminal_bonus": float(terminal_bonus),
        "total_reward": float(total_reward),
    }


def compute_reward(
    *,
    scenario: str,
    prev_distance: float,
    cur_distance: float,
    action: tuple[float, float, float],
    evader_position: tuple[float, float, float],
    pursuer_position: tuple[float, float, float],
    bounds: tuple[float, float, float, float, float, float],
    outcome: EpisodeOutcome,
) -> float:
    terms = compute_reward_terms(
        scenario=scenario,
        prev_distance=prev_distance,
        cur_distance=cur_distance,
        action=action,
        evader_position=evader_position,
        pursuer_position=pursuer_position,
        bounds=bounds,
        outcome=outcome,
    )
    return float(terms["total_reward"])
