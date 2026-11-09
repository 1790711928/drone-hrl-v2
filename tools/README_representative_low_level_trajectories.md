# Representative low-level trajectories

This tool selects one successful fixed-seed trajectory for each of `pi1_rear`,
`pi2_flank`, `pi3_boundary`, and `pi4_vertical`. It uses trajectory geometry
only; it does not call the score matrix or heatmap code.

Run locally after placing the four private SAC checkpoints in
`outputs/checkpoints`:

```powershell
python tools/select_representative_low_level_trajectories.py `
  --checkpoint-dir outputs/checkpoints `
  --seed-start 0 `
  --seed-end 199 `
  --output-dir outputs/representative_low_level_trajectories `
  --deterministic `
  --plot-3d `
  --plot-xoy `
  --add-aircraft
```

Checkpoint overrides are available with `--pi1-rear-checkpoint`,
`--pi2-flank-checkpoint`, `--pi3-boundary-checkpoint`, and
`--pi4-vertical-checkpoint`. The default filenames are the four files named in
the main repository checkpoint convention.

The tool scans only successful, finite, complete episodes. It ranks the four
policies independently: pi1 rewards forward-axis and distance expansion while
penalizing vertical displacement; pi2 rewards right-axis excursion and lateral
path length; pi3 rewards early boundary approach followed by margin recovery;
pi4 rewards vertical ascent while penalizing horizontal displacement.

Each selected figure is a two-view PNG containing a 3D view and an XOY view.
The evader is red with a solid line and the pursuer is blue with a dashed line.
The endpoint glyph is a small four-rotor aircraft oriented by the final valid
velocity direction. The command creates CSV/JSON/PNG outputs only when run
locally with available checkpoints.
