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
