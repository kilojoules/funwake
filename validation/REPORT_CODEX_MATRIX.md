# Codex full-optimizer matrix eval (Task P)

3 codex full-opt champions (`optimize()`, train 5583-5584 GWh on DEI N=50)
scored across the 64-cell matrix (2 polygons × 4 roses × 8 N). Answers whether
the richer `optimize()` interface generalizes, or only shatters the DEI
training ceiling.

## Headline

The full-optimizer interface breaks the DEI training ceiling (+40 GWh at
N=50) but that headroom does **not** translate into matrix-wide superiority
or held-out superiority, and it **loses the scaling property**. The
schedule-only dual-bump is more robust.

## Vs per-cell 500-multistart baseline (48 cells with baselines, N≤80)

- Beats baseline on **34/48 (71%)**, mean **+13.3 GWh**.
- **By rose**: dei 11/12 (+32.1), omnidir 11/12 (+23.8), rowp 12/12 (+12.3),
  **uniform 0/12 (−14.9)**.
- The uniform-rose cells are a **universal failure** — every codex champion
  loses to baseline there, exactly as the schedule-only schedules and the
  baseline's own multistart do. Uniform wind is a landscape property, not an
  interface artifact.

## Vs schedule-only Claude dual-bump (per cell)

- Codex full-opt > dual-bump on only **25/45 (56%)** — a coin flip.
- Despite +40 GWh training headroom at N=50, the full-opt interface does
  **not** systematically beat the schedule-only schedule across the matrix.

## Scaling: full-opt does NOT reach high-N

- All **43 errors are N=200/300 timeouts** (600s cap). 19 at N=200, 24 at
  N=300. Only run1 (the simple 63-line champion) occasionally completes one
  high-N cell; run2/run3 (476/568-line multi-start+BO+ensemble) always time
  out.
- Schedule-only schedules run N=200/300 fine (schedule matrix + high-N sweep
  completed). So the schedule-only interface **scales where full-opt cannot**.

## Held-out (ROWP, from Task N)

- Codex champions on ROWP: 4233.5 / 4258.9 / 4235.8 GWh — all feasible, all
  **below** deployed dual-bump 4271.5 and below Gemini full-opt 4272.7.
- Combined with the ~4272 GWh ceiling reached from both interfaces and never
  exceeded, the held-out band is saturated.

## Paper implication — "nothing great left on the table"

Three independent bounds now agree that the ~4272 GWh held-out optimum is a
real landscape ceiling for the DEI→ROWP pair:
1. Random search in the dual-bump family: 4268.6 (within 4 GWh).
2. Schedule-only replicas (3 distinct families): 4250-4260.
3. Full-optimizer interface (codex): shatters DEI training (+40) but held-out
   4234-4259, matrix-superiority a coin flip (56%), and does not scale to
   high-N.

The training-side headroom the full-optimizer interface exposes is a mirage:
it over-fits DEI without transferring. The schedule-only dual-bump is the more
robust deployable object (scales to N=300, matches full-opt on ~44% of matrix
cells at a fraction of the compute). Real headroom lives only in the
uniform-rose regime (a universal failure across all interfaces and the
baseline) — that is the follow-up-paper direction, not a within-interface win.

## Files
- `results/matrix/codex_fullopt_matrix.json` — 192 cell results (149 ok, 43 timeout)
- `tools/eval_matrix_fullopt.py` — runner
- `tools/analyze_codex_matrix.py` — this analysis
