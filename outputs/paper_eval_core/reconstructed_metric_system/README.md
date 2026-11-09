# Reconstructed metric system

This directory is independent of all existing evaluation scripts and outputs.
It implements the requested four-policy × four-scenario behavior score without
generating a heatmap.

## Files

- `metric_design_spec.md`: concepts, program mappings, and explicit assumptions.
- `metric_config.yaml`: scenarios, fixed normalization bounds, validity, and
  aggregation protocol.
- `eval_reconstructed_behavior.py`: paired deterministic trajectory evaluator.
- `score_reconstructed_matrix.py`: normalization, `V`, per-episode score, and
  4×4 CSV summaries.
- `reproduce_historical_directional_score.py`: reproduces a score matrix from
  the saved 2026-10-08 directional raw CSV without rerunning models.
- `plot_historical_score_heatmap.py`: creates an annotated heatmap only after
  the score matrix has passed review.
- `reports/`: reserved for run-specific reports; no existing output directory
  is used.

`metric_config.yaml` is written in JSON-compatible YAML. The scripts use
PyYAML when available and automatically fall back to Python's standard-library
JSON parser, so the project virtual environment does not need an extra package.

## Run order

Run from the repository root with the project virtual environment:

```powershell
$repo = "E:\python程序\drone_hrl_repo\drone-hrl-v2"
$py = "$repo\.venv\Scripts\python.exe"
$recon = "$repo\outputs\paper_eval_core\reconstructed_metric_system"
$out = "$recon\runs\baseline_20261025"

& $py "$recon\eval_reconstructed_behavior.py" `
  --config "$recon\metric_config.yaml" `
  --output-dir "$out" `
  --episodes 100 `
  --seed 20261025 `
  --policy1-checkpoint "<path-to-pi1-model.zip>" `
  --policy2-checkpoint "<path-to-pi2-model.zip>" `
  --policy3-checkpoint "<path-to-pi3-model.zip>" `
  --policy4-checkpoint "<path-to-pi4-model.zip>" `
  --device cpu

& $py "$recon\score_reconstructed_matrix.py" `
  --config "$recon\metric_config.yaml" `
  --input-csv "$out\episodes_raw.csv" `
  --output-dir "$out\scored"
```

The evaluator refuses to write into a non-empty output directory. Use a new
run directory for every parameter change. The scripts do not know or alter the
locations of existing models; model paths are supplied explicitly.

To score the saved historical directional export, use a new output directory:

```powershell
& $py "$recon\reproduce_historical_directional_score.py" `
  --input-csv "$repo\outputs\paper_eval_core\final_directional_diag\behavior_metrics_raw.csv" `
  --output-dir "$recon\runs\historical_final_directional_score_v1"
```

This command uses the saved raw data, the directional metric pairs, global
observed min–max normalization, and the saved `task_valid` field. It writes a
reviewable matrix and report but does not create a heatmap.

After reviewing `historical_score_report.md` and confirming the diagonal check,
create the figure with:

```powershell
& $py "$recon\plot_historical_score_heatmap.py" `
  --matrix-csv "$recon\runs\historical_final_directional_score_v1\historical_score_matrix.csv" `
  --output-png "$recon\runs\historical_final_directional_score_v1\historical_behavior_score_heatmap.png"
```

For the review-only contrast-enhanced display (the raw score matrix is not
changed), use the softmax plotter. `--beta 8` increases within-scenario
separation and `--scale 100` displays the relative values as 0--100 scores:

```powershell
& $py "$recon\plot_historical_score_softmax.py" `
  --matrix-csv "$recon\frozen_historical_directional_score_v1\historical_score_matrix.csv" `
  --output-png "$recon\frozen_historical_directional_score_v1\historical_behavior_score_heatmap_softmax_beta8_x100.png" `
  --output-csv "$recon\frozen_historical_directional_score_v1\historical_score_softmax_beta8_x100.csv" `
  --beta 8 `
  --scale 100
```

## Outputs

`episodes_raw.csv` contains one row per policy/scenario/episode, including the
initial-state seed, outcome, all eight raw indicators, and trace summary fields.
The scoring script writes `episodes_scored.csv`, `metric_summary.csv`,
`score_matrix.csv`, and `score_run_manifest.json`. No heatmap is produced.

## Local checks still required

The fixed bounds and `escaped_only` validity factor are reconstruction
assumptions. After a local run, inspect raw ranges and outcome counts before
deciding whether a new configuration version is justified. Do not tune bounds
per policy or overwrite a previous run.
