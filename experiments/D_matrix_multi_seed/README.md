# Experiment D — Per-seed 64-cell evaluation matrix

## Reviewer concern
"Uniform-wind failure mode still gets one sentence. Given you now have
seed variance, it would be trivial to report whether uniform-wind failure
is consistent across seeds or seed-dependent."

Also: Fig 3 currently shows a single-seed snapshot of the 64-cell matrix.
For a paper that argues schedule generalization, error bars are warranted.

## Hypothesis under test
The uniform-wind cell is a *systematic* failure (consistent across init
seeds), not noise.

## Method
Re-evaluate the top Claude (`iter_192`) and top Gemini (`iter_118`)
schedules across all 64 cells, with 3 init seeds per cell. Mean and
range per cell. Specifically inspect uniform-wind cells: if std < 0.1%
of baseline, declare systematic; if > 0.5%, seed-dependent.

## Cost
2 schedules × 64 cells × 3 seeds = 384 evals.
Time per eval ranges 30s (N=30) to 340s (N=200) and ≈900s (N=300).
Mean ~120s. Total ~13 hr CPU. Embarrassingly parallel; on LUMI
with 20 CPU jobs ≈ 40 min.

## Inputs
- `results/matrix/manifest.json`
- Top schedules (auto-discovered from
  `results_agent_schedule_only_5hr/iter_192.py` and
  `results_agent_gemini_cli_5hr/iter_118.py`)

## Outputs
- `experiments/D_matrix_multi_seed/results.json`
  - per (script, cell, seed): aep_gwh, feasible, time_s
  - per (script, cell): mean, std, n, feasible_fraction

## Success criteria
- Uniform-wind cells show feasible_fraction ∈ {0, 1} consistently across
  seeds (binary, systematic). If fraction is intermediate, seed-noise
  is the failure driver and the paper must be re-framed.
- Per-cell std on the OTHER cells should be small (<0.5% of baseline)
  for the headline "schedules generalize" claim.

## Launch (local)
```
bash experiments/D_matrix_multi_seed/launch_local.sh
```

## Launch (LUMI parallel)
```
sbatch experiments/D_matrix_multi_seed/run_matrix_multi_seed.sbatch
```

## Aggregation
```
pixi run python experiments/D_matrix_multi_seed/aggregate.py
```
Produces:
- `experiments/D_matrix_multi_seed/summary.json`
- Updated `paper/figs/fig_short_matrix.png` with error bars (run
  `paper/make_figures.py` afterwards).
