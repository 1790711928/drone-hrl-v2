# Reconstructed four-policy × four-scenario behavior score

Status labels used in this document:

- `CONFIRMED_CONCEPT`: fixed by the requested Chapter 3 design.
- `IMPLEMENTATION_MAPPING`: a concrete program field/trace is used to express
  the concept in this reconstruction.
- `ASSUMPTION`: a missing historical detail that is explicit in
  `metric_config.yaml` and must be checked against local data.

This is a new, isolated reconstruction. It is not a claim about the missing
historical implementation and it does not import ESAS, `skill_alignment`, or a
previous score-matrix script.

## Evaluation protocol

Every policy is evaluated in every scenario. One deterministic initial state is
sampled per `(scenario, episode, seed)` and copied to all four policies in that
scenario. All policies use deterministic model actions, the same environment
termination configuration, and the same episode length implied by that
configuration. Raw traces are retained before scoring.

The matrix is indexed as rows = policies (`pi1`–`pi4`) and columns = scenarios
(`S1`–`S4`). The intended diagonal property is a diagnostic target, not an
assumption inserted into the score.

## Indicator definitions

| Indicator | Status | Program realization in this reconstruction |
|---|---|---|
| `M_turn` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Mean of `abs(threat_right) * max(-threat_forward, 0)` over the trace. The local threat frame supplies the two projections. The exact historical window/scale was missing, so the full trace and fixed normalization bounds are explicit assumptions. |
| `G_v` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Late-window mean evader speed minus early-window mean evader speed. This is deliberately a late-vs-early acceleration proxy, not ordinary mean speed. The 20% windows and physical speed bounds are assumptions. |
| `G_s` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Late minus early signed lateral gap in a fixed initial evader frame. The fixed frame, 20% windows, and initial-distance scaling are explicit reconstruction assumptions. |
| `R_s` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Fraction of adjacent lateral-gap samples that do not decrease; the adjacent-sample convention is an explicit reconstruction assumption. |
| `G_b` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Final normalized minimum boundary margin minus initial normalized minimum boundary margin; use of the environment-normalized margin is an explicit mapping assumption. |
| `R_b` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Fraction of trace samples with normalized minimum boundary margin at least `0.35`. The threshold is explicit and must be validated for this reconstruction. |
| `G_z` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Final absolute vertical separation minus initial absolute vertical separation; the absolute-separation convention is an explicit mapping assumption. |
| `G_d` | `CONFIRMED_CONCEPT` + `IMPLEMENTATION_MAPPING` + `ASSUMPTION` | Final 3-D pursuer–evader distance minus initial 3-D distance; terminal-minus-initial aggregation is an explicit mapping assumption. |

## Normalization

For every indicator, the same fixed bounds are applied to every policy and
scenario:

```text
m_bar = clip((raw - lower) / (upper - lower), 0, 1)
```

The bounds in `metric_config.yaml` are `ASSUMPTION`, not learned from the
evaluated policies. Per-policy min/max fitting is prohibited because it would
destroy cross-policy comparison.

## Validity factor and aggregation

`V=1` only when the environment reports `outcome == "escaped"`; captured,
out-of-bounds, and timeout episodes receive `V=0`. This is an explicit
`ASSUMPTION` for the reconstruction and is not presented as recovered code.

For the scenario's designated metric pair `(m1, m2)`:

```text
S_episode = V * ((m1_bar + m2_bar) / 2)
```

The 4×4 cell value is the arithmetic mean of `S_episode` over episodes. Raw
metrics, normalized metrics, `V`, and the per-episode score are all saved.

## Required review after local execution

Before accepting a matrix, inspect the raw distributions, saturation at the
fixed normalization bounds, outcome counts, and paired-state reproducibility.
If the fixed bounds or `V` convention are changed, record the change as a new
configuration version; do not overwrite an earlier run.
