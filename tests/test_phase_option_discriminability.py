import numpy as np

from src.evaluation.eval_phase_option_discriminability import (
    evaluate_phase_option,
    expand_eval_modes,
    parse_phase_types,
)
from src.training.highlevel_env import HighLevelOptionEnv


class CountingZeroModel:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, obs, deterministic=True):
        self.predict_calls += 1
        return np.zeros(3, dtype=np.float32), None


def test_parse_phase_types_supports_all_and_csv_subset():
    assert parse_phase_types("all") == ["rear", "flank", "boundary", "vertical", "rear_vertical"]
    assert parse_phase_types("rear,vertical") == ["rear", "vertical"]


def test_expand_eval_modes_supports_both():
    assert expand_eval_modes("one_shot") == ["one_shot"]
    assert expand_eval_modes("sustained") == ["sustained"]
    assert expand_eval_modes("both") == ["one_shot", "sustained"]


def test_one_shot_executes_only_one_option_window():
    models = [CountingZeroModel() for _ in range(4)]
    env = HighLevelOptionEnv(models, option_duration=2, scenario_set="sequential")
    row = evaluate_phase_option(env, "rear", option_index=0, episodes=1, eval_mode="one_shot")
    assert models[0].predict_calls == 2
    assert row["eval_mode"] == "one_shot"
    assert row["sustained_success_rate"] == ""


def test_sustained_repeats_option_windows_until_terminal_result():
    models = [CountingZeroModel() for _ in range(4)]
    env = HighLevelOptionEnv(models, option_duration=2, scenario_set="sequential")
    row = evaluate_phase_option(env, "rear", option_index=0, episodes=1, eval_mode="sustained")
    assert models[0].predict_calls > 2
    assert row["eval_mode"] == "sustained"
    assert row["one_shot_success_rate"] == ""


def test_flank_score_is_driven_by_lateral_reduction_not_distance_gain():
    from src.evaluation.eval_phase_option_discriminability import _improvement_metrics

    start = {"distance": 0.10, "closing_speed": 0.10, "threat_forward": -0.5, "threat_right": 0.8, "min_boundary_margin": 0.3, "dz": 0.2, "boundary_margin_z": 0.3, "threat_up": 0.0}
    lateral = dict(start, threat_right=0.4)
    distance_only = dict(start, distance=0.30)
    assert _improvement_metrics("flank", start, lateral)["improvement_score"] > _improvement_metrics("flank", start, distance_only)["improvement_score"]


def test_canonical_and_injected_geometry_can_be_reset_without_models():
    from src.evaluation.eval_phase_canonical_alignment import geometry_row

    env = HighLevelOptionEnv([None] * 4, option_duration=2, scenario_set="sequential")
    canonical = geometry_row(env, "flank", "canonical")
    injected = geometry_row(env, "flank", "injected")
    assert canonical["canonical_scenario"] == "flank_threat"
    assert abs(canonical["threat_right"] - injected["threat_right"]) < 1e-9
    assert abs(canonical["threat_forward"] - injected["threat_forward"]) < 1e-9
