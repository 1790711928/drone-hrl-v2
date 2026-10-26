# Frozen historical directional behavior score v1

This freeze is derived from the saved 2026-10-08
`final_directional_diag/behavior_metrics_raw.csv` export (80 episode rows,
five episodes per policy–scenario cell).

The score uses the historical directional fields:

- S1: `rear_gap_gain` + `rear_gap_non_decreasing_ratio`
- S2: `side_gap_gain` + `side_gap_non_decreasing_ratio`
- S3: `boundary_margin_gain` + `safe_boundary_ratio`
- S4: `vertical_away_displacement_gain` + `distance_gain`

Each metric is normalized using one global observed min–max range across all 80
rows. `V` is the saved `task_valid` field. The cell score is the mean of the
per-episode values `V * (m1_bar + m2_bar) / 2`.

The resulting matrix has all four target policies highest in their matching
scenario. The PNG is generated from the saved matrix and contains the numeric
cell values; it is a frozen candidate for the paper heatmap.

This directory contains derived results only. The original historical raw CSV
and workbook remain in the user's local `final_directional_diag` directory.
