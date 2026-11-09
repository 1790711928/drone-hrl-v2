from pathlib import Path

from src.evaluation.plot_highlevel_win_rate_curves import _read_series, plot_win_rate_curves


def test_read_series_sorts_by_training_step(tmp_path: Path):
    csv_path = tmp_path / "win_rate.csv"
    csv_path.write_text(
        "checkpoint_step,win_rate\n200,0.8\n100,0.6\n",
        encoding="utf-8",
    )
    assert _read_series(f"complete={csv_path}") == ("complete", [100, 200], [60.0, 80.0])


def test_plot_win_rate_curves_writes_line_plot(tmp_path: Path):
    csv_path = tmp_path / "win_rate.csv"
    csv_path.write_text(
        "checkpoint_step,win_rate\n100,0.6\n200,0.8\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "comparison.png"
    plot_win_rate_curves([f"complete={csv_path}"], output_path, "test")
    assert output_path.exists()
    assert output_path.stat().st_size > 0
