# Low-level trajectory candidate scanner

This tool scans fixed-seed rollouts from the four private SAC checkpoints and
keeps every successful, finite trajectory for manual review. It does not call
the behavior-score or heatmap pipelines and does not automatically choose the
final paper trajectory. It records pursuer-process diagnostics including
distance by step, minimum distance, time to minimum, initial closing trend,
pursuit duration, pursuer path length, and final relative speed.

Run locally after placing the four private SAC checkpoints in
`outputs/checkpoints`:

```powershell
python tools/select_representative_low_level_trajectories.py `
  --checkpoint-dir outputs/checkpoints `
  --seed-start 0 `
  --seed-end 199 `
  --output-dir outputs/representative_low_level_trajectories `
  --deterministic
```

Checkpoint overrides are available with `--pi1-rear-checkpoint`,
`--pi2-flank-checkpoint`, `--pi3-boundary-checkpoint`, and
`--pi4-vertical-checkpoint`. The default filenames are the four files named in
the main repository checkpoint convention.

The aggregate table is `candidate_trajectories_metrics.csv`; each successful
rollout is written as a complete per-step CSV under `trajectories/`. The
`candidate_rejection_reason` column explains why a rollout was excluded from
the manual candidate set. Ranking scores are supplied only as a review aid;
there is no automatic final seed selection.

The retained plotting function produces a single 3D view only. The evader is
red with a solid line and the pursuer is blue with a dashed line; both starts
are black `x` markers. XOY views and endpoint aircraft glyphs are disabled.
Use `--plot-3d` only when you explicitly want review PNGs after the candidate
table has been inspected. The command creates outputs only locally with
available checkpoints.
