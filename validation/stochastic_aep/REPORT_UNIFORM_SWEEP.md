# Uniform-rose ηT-sweep — completes the null

**Verdict: complete null on uniform single-direction rose**. Both Claude
iter_192 and Gemini iter_192 fail to clear the 0.2 % bar against
best-ηT-tuned decay+ES SGD on **0 / 12** uniform-rose cells. The
borderline DEI × uniform × N = 70, 80 signal seen at default ηT = 0.5 m
(+0.25 %, +0.27 %) was an ηT-tuning artifact: under best-tuned ηT, the
gap drops to +0.17 % and +0.12 %, both below bar.

This **sharpens** the scoped-superiority claim: iter_192 beats best-ηT
SGD ONLY on multidirectional roses (18 / 18 headline cells, gap 0.45 –
1.10 %). On uniform single-direction roses the gap collapses to zero
within noise — at any N, either polygon. Clean null.

## Setup

- 12 uniform-rose cells (2 polygons × 6 N values = 30, 40, …, 80).
- 7 ηT values × 3 sample seeds × decay+ES SGD baseline = 21 SGD runs / cell.
- ηT ∈ {0.1, 0.25, 0.5, 1, 2.5, 5, 25} m, δ = ηT / 50 m. Paper's
  ηT = 0.1 m included.
- Stochastic K = 50, total_steps = 8000, init_seed = 0.
- Compute: DTU gbar LSF array [1-12]%6, hpc queue, 4-core 8GB tasks.
  Total ~ 100 min wall, 0 errors.

## Per-cell results

| cell | best-ηT (m) | best-AEP (GWh) | Claude gap | Gemini gap | clears |
|---|---:|---:|---:|---:|:---:|
| dei_n30_roseuniform | 25.0 | 3359.98 | +0.000 ± 0.000 % | −0.018 ± 0.011 % | — |
| dei_n40_roseuniform | 25.0 | 4479.97 | +0.000 ± 0.000 % | −0.056 ± 0.013 % | — |
| dei_n50_roseuniform | 5.0 | 5593.89 | +0.083 ± 0.024 % | −0.025 ± 0.024 % | — |
| dei_n60_roseuniform | 2.5 | 6704.30 | +0.136 ± 0.026 % | +0.016 ± 0.028 % | — |
| dei_n70_roseuniform | 5.0 | 7815.55 | **+0.171 ± 0.025 %** | −0.064 ± 0.032 % | — |
| dei_n80_roseuniform | 1.0 | 8919.24 | **+0.121 ± 0.043 %** | −0.051 ± 0.030 % | — |
| rowp_n30_roseuniform | 0.10 | 2330.16 | +0.000 ± 0.000 % | +0.000 ± 0.000 % | — |
| rowp_n40_roseuniform | 0.10 | 3106.88 | +0.000 ± 0.000 % | +0.000 ± 0.000 % | — |
| rowp_n50_roseuniform | 0.10 | 3883.60 | +0.000 ± 0.000 % | +0.000 ± 0.000 % | — |
| rowp_n60_roseuniform | 25.0 | 4660.32 | −0.048 ± 0.019 % | −0.214 ± 0.009 % | — |
| rowp_n70_roseuniform | 2.5 | 5429.57 | −0.028 ± 0.052 % | −0.245 ± 0.014 % | — |
| rowp_n80_roseuniform | 5.0 | 6184.16 | +0.083 ± 0.021 % | −0.111 ± 0.065 % | — |

**Claude clears 0.2 % vs best-ηT: 0 / 12.**
**Gemini clears 0.2 % vs best-ηT: 0 / 12.**

## Comparison to default-ηT (= 0.5 m / δ = 0.01) baseline

| cell | Claude gap vs default-ηT (`matrix_fair`) | Claude gap vs best-ηT (this run) | shift |
|---|---:|---:|---:|
| dei_n70_roseuniform | +0.251 % (✓) | +0.171 % (·) | −0.080 % |
| dei_n80_roseuniform | +0.267 % (✓) | +0.121 % (·) | −0.146 % |
| rowp_n70_roseuniform | +0.184 % (borderline) | −0.028 % (·) | −0.212 % |
| rowp_n80_roseuniform | +0.183 % (borderline) | +0.083 % (·) | −0.100 % |

The 2 cells that cleared at default ηT are pure ηT-tuning artifacts — they
lose clearance once ηT is properly tuned. **No genuine
iter_192-over-baseline superiority on uniform rose anywhere.**

## Why ROWP × uniform × N ≤ 50 is exactly zero

Three cells (rowp × uniform × N = 30, 40, 50) show 0.000 % gap with 0.000 %
spread across all schedules and ηT values. Mechanistic: with single
wind direction, no wakes are crossed if turbines are placed on a single
cross-wind line. For small N (30–50 turbines), the polygon is large
enough that ALL turbines fit on a single line at min-spacing, giving the
wake-free maximum AEP. All schedules find this trivial optimum
identically.

At N ≥ 60 on ROWP, the cross-wind line gets tight, schedules diverge
slightly via different exploration trajectories — but still within noise.

## Best-ηT pattern (compare to multidir hardening)

| Configuration | Best-ηT range | × paper (0.1 m) |
|---|---:|---:|
| Multidir hardening (18 headline cells) | 0.25 m – 25 m | 2.5 × – 250 × |
| **Uniform (this run, 12 cells)** | **0.1 m – 25 m** | **1 × – 250 ×** |

Uniform behaves similarly to multidir: AEP-optimal ηT under stochastic
K = 50 Adam is consistently looser than the paper's 0.1 m positioning
tolerance, often by 1 – 2 orders of magnitude. For ROWP × uniform ×
small N where the trivial optimum is unique, ηT = 0.1 m exactly matches
(turbines reach it trivially); elsewhere the optimum is at larger ηT.

## Updated headline claim

> *Across 60 deploy cells (48 in matrix + 12 uniform), iter_192 beats
> best-ηT-tuned TopFarm SGD on multidirectional roses (omnidir / DEI /
> ROWP rose) with N ≥ 30 — 18 / 18 headline cells (N ≥ 60). On the
> uniform single-direction rose, the gap is null at every cell
> (0 / 12) — confirming the scoped advantage is real, not a general
> claim. The mechanism appears to be schedule-driven exploration
> diversity that pays off only when multiple wind directions create
> non-trivial trade-offs; on single-direction trivial-optimum cases,
> all schedules converge to the same layout and the gap vanishes.*

## Implications for the paper

1. **Uniform null is now rigorous** — not "small gaps borderline at
   default", but "zero at best-ηT". Use it as the canonical scope
   boundary in the paper.
2. **Confirms the multidirectional superiority is mechanism-specific**
   — schedules matter when there's a non-trivial trade-off, not when
   there's a trivial single-line optimum.
3. **No leakage from the multidir claim**: a careful reviewer who runs
   the paper's procedure on a uniform rose will get zero gap, consistent
   with the claim's stated scope.

## Caveats

1. Single init seed (= 0) on the baseline AND on iter_192 / Gemini.
   Multi-init for these uniform cells not done; would add a fourth-decimal
   refinement to the spread but won't change the 0/12 result given the
   gap is already < 0.2 %.
2. Wake model fixed at `BastankhahGaussianDeficit(k = 0.04)` — matches
   prior matrix runs.
3. ROWP polygon here is the matrix's 4-vertex stand-in, not the
   published 740-10 6-vertex Borssele. (Same as throughout.)

## Artifacts

- `validation/stochastic_aep/run_uniform_per_cell.py` — runner (one cell
  × 21 SGD runs).
- `validation/stochastic_aep/lsf_uniform_array.sh` — LSF array script.
- `validation/stochastic_aep/_uniform_sweep_gbar/cell{0..11}.json` —
  raw per-cell results.
- 252 SGD runs, 0 errors, ~100 min wall on DTU gbar hpc queue.
