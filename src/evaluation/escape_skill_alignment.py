from __future__ import annotations

from collections.abc import Mapping

ESAS_COMPONENTS = {
    "rear": [
        "rear_distance_gain_component",
        "rear_closing_speed_component",
        "rear_safety_component",
        "rear_direction_consistency_component",
    ],
    "flank": [
        "flank_threat_right_component",
        "flank_lateral_component",
        "flank_distance_component",
        "flank_safety_component",
    ],
    "boundary": [
        "boundary_margin_improvement_component",
        "boundary_final_margin_component",
        "boundary_safety_component",
    ],
    "vertical": [
        "vertical_separation_component",
        "vertical_threat_up_component",
        "vertical_z_safety_component",
        "vertical_distance_component",
    ],
}

ESAS_WEIGHTS = {
    "rear": {
        "rear_distance_gain_component": 0.25,
        "rear_closing_speed_component": 0.25,
        "rear_safety_component": 0.15,
        "rear_direction_consistency_component": 0.35,
    },
    "flank": {
        "flank_threat_right_component": 0.30,
        "flank_lateral_component": 0.30,
        "flank_distance_component": 0.20,
        "flank_safety_component": 0.20,
    },
    "boundary": {
        "boundary_margin_improvement_component": 0.40,
        "boundary_final_margin_component": 0.35,
        "boundary_safety_component": 0.25,
    },
    "vertical": {
        "vertical_separation_component": 0.35,
        "vertical_threat_up_component": 0.25,
        "vertical_z_safety_component": 0.20,
        "vertical_distance_component": 0.20,
    },
}

SCENARIO_TO_ESAS = {
    "rear_close_threat": "rear",
    "flank_threat": "flank",
    "boundary_constrained": "boundary",
    "vertical_z_threat": "vertical",
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_score(skill: str, components: Mapping[str, float]) -> float:
    weights = ESAS_WEIGHTS[skill]
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"ESAS weights for {skill!r} must sum to 1.0, got {weight_sum}")
    return _clip01(sum(weights[name] * _clip01(float(components.get(name, 0.0))) for name in weights))


def compute_rear_esas(components: Mapping[str, float]) -> float:
    return _weighted_score("rear", components)


def compute_flank_esas(components: Mapping[str, float]) -> float:
    return _weighted_score("flank", components)


def compute_boundary_esas(components: Mapping[str, float]) -> float:
    return _weighted_score("boundary", components)


def compute_vertical_esas(components: Mapping[str, float]) -> float:
    return _weighted_score("vertical", components)


def compute_all_esas_scores(components: Mapping[str, float]) -> dict[str, float]:
    return {
        "rear_esas": compute_rear_esas(components),
        "flank_esas": compute_flank_esas(components),
        "boundary_esas": compute_boundary_esas(components),
        "vertical_esas": compute_vertical_esas(components),
    }


def compute_esas_for_scenario(scenario: str, components: Mapping[str, float]) -> float:
    skill = SCENARIO_TO_ESAS[scenario]
    return compute_all_esas_scores(components)[f"{skill}_esas"]
