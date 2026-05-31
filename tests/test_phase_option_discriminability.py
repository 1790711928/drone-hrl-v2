from src.evaluation.eval_phase_option_discriminability import parse_phase_types


def test_parse_phase_types_supports_all_and_csv_subset():
    assert parse_phase_types("all") == ["rear", "flank", "boundary", "vertical", "rear_vertical"]
    assert parse_phase_types("rear,vertical") == ["rear", "vertical"]
