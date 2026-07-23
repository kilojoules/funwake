# Equivalent-cost SGD frontier (budget = 8000 gradient evaluations)

Topfarm-style SGD with Quick-2023 early stopping, K vmapped multistarts of
T = 8000//K iterations each (evals billed = K*T = 8000, matching one
schedule-mode run). Tool: `tools/run_budgeted_baseline.py`. Seeds 0-9 per
config; 100 DEI runs + 10 ROWP runs, zero failures. Total solver time 3605 s.

## DEI frontier (results/problem_dei_n50.json, 50 turbines)

| Config (K x T) | AEP mean +/- std (GWh) | Best AEP (GWh) | Feasible runs | Mean n_feasible/K | Mean evals executed | Mean time (s) |
|---|---|---|---|---|---|---|
| 1 x 8000 | 5523.6 +/- 6.4 | 5531.5 | 8/10 | 0.80 | 7345 | 28.5 |
| 2 x 4000 | 5527.2 +/- 2.6 | 5531.4 | 10/10 | 1.00 | 7347 | 33.2 |
| 3 x 2666 | 5526.8 +/- 2.8 | 5530.5 | 10/10 | 0.87 | 7348 | 40.4 |
| 4 x 2000 | 5527.6 +/- 5.5 | 5538.3 | 10/10 | 0.97 | 7351 | 45.7 |
| 6 x 1333 | 5529.0 +/- 4.9 | 5537.9 | 10/10 | 0.98 | 7356 | 48.7 |
| **8 x 1000** | **5530.3 +/- 3.1** | 5535.5 | 10/10 | 0.95 | 7369 | 35.6 |
| 12 x 666 | 5529.5 +/- 3.0 | 5533.6 | 10/10 | 0.98 | 7369 | 33.8 |
| 16 x 500 | 5530.0 +/- 2.6 | 5533.6 | 10/10 | 0.99 | 7395 | 29.4 |
| 24 x 333 | 5530.6 +/- 4.0 | 5537.4 | 10/10 | 0.97 | 7426 | 20.2 |
| 32 x 250 | 5530.2 +/- 3.9 | 5536.8 | 10/10 | 0.97 | 7434 | 21.0 |

Winner (highest mean best-feasible AEP): **K=24, T=333** at 5530.6 GWh mean --
but its edge over K=8 (+0.25 GWh) is **below the 0.3 GWh config-mean noise
floor**, so K=8..32 are statistically tied and **K=8, T=1000 remains the
reference config** (ROWP generalization was not rerun).

**No turnover through K=32**: the frontier rises from K=1 (5523.6) to K=8
(5530.3) and then plateaus (K=8..32 means span 5529.5-5530.6 GWh, all within
noise of each other). No AEP cliff and no feasibility collapse at small T
(even T=250: 10/10 feasible, mean n_feasible/K >= 0.95). Breadth stops paying
beyond K~8 but does not yet hurt.

Reference points:
- **DEI 500-multistart baseline: 5540.7 GWh** (best-of-500 x 6000 iters ~ 3M
  evals -- roughly **375x** the cost of these runs).
- **Seed schedule at the same 8000-eval budget: 5529.2 GWh but INFEASIBLE**
  (boundary violation, per a recent scorer run).

## ROWP generalization (results/problem_rowp.json, 74 turbines, held out)

Reference config K=8, T=1000, seeds 0-9 (kept as reference: no extended-
frontier config beat K=8's mean by more than the 0.3 GWh noise floor, so
ROWP was not rerun):

| Metric | Value |
|---|---|
| AEP mean +/- std | 4228.5 +/- 3.3 GWh |
| AEP min / max | 4224.2 / 4231.9 GWh |
| Feasible runs | 10/10 (mean n_feasible/K = 0.98) |
| Mean evals executed / billed | 7432 / 8000 |
| Mean time per run | 24.0 s |
| Baseline (`aep_gwh` field of `results/baseline_rowp.json`, best of 500 multistarts) | 4246.7 GWh |
| Gap to baseline (mean / best) | 18.2 / 14.8 GWh (0.43% / 0.35%) |

## Early-stopping calibration (results/equiv_cost_sgd/es_calibration.json)

Probe context: TopFarm-style 6000-iter run (4000 decay + 2000 constant-lr),
DEI, 10 seeds. Headline numbers:
- Activation: analytic prediction step 4079; first ES-active iteration 4080
  (~52% of the decay phase in this setting; in the budgeted tool activation
  lands at ~0.916*T).
- Tail spread after activation: 2-4 iterations (median 2); evals 4082-4084 of
  6001, i.e. ES saves ~32% of iterations.
- Feasibility rate with ES: 10/10.
- ES-vs-full AEP delta: **AEP-neutral**. The calibration file's +0.415 GWh
  figure comes from only 2 paired seeds; the 10-seed ES-mechanism study
  (`results/equiv_cost_sgd/es_mechanism/REPORT.md`) refutes any AEP increase:
  mean delta -0.007 GWh, std 0.45, 5 positive / 5 negative. The correct
  statement is that ES saves ~32% of iterations and restores exact feasibility
  at zero expected AEP cost.

## Noise floor caveat

The AEP objective on this stack is piecewise-constant at mm scale (hard
wake-cone mask in `pixwake/deficit/base.py`), giving single-evaluation
"texture" noise of ~0.25-0.3 GWh std (~+/-0.5 GWh swings). Any single-run AEP
difference below ~0.5 GWh is texture-dominated, and with 10 seeds per config,
config-mean differences below ~0.3 GWh are statistically indistinguishable.
Per-seed rankings within that margin are noise.

## Interpretation

The frontier is remarkably flat but tilts toward breadth before saturating:
mean AEP rises from 5523.6 GWh (K=1) to ~5530 GWh at K=8, then plateaus --
K=8, 12, 16, 24, 32 all land in a 5529.5-5530.6 GWh band whose internal
differences are below the ~0.3 GWh config-mean noise floor. K=1 is the only
config that ever fails feasibility (8/10), because a single lane has no
fallback when its one trajectory ends infeasible. The K>=8 vs K=1-3 mean
differences (~3-7 GWh) exceed the noise floor, so the breadth advantage is
real, but past K~8 additional restarts buy nothing measurable: even 250
iterations (with the cramped C=round(0.45*T)=112, M=round(0.9*T)=225
schedule) still converges each lane to a good, feasible local optimum
(10/10 feasible, mean n_feasible/K >= 0.95 through K=32), so the anticipated
depth-starvation turnover has not appeared by K=32. Wall-clock cost per run
actually drops at high K (20-21 s at K=24-32 vs 36-49 s at K=4-8) because
shorter while-loops dominate the vmap width on this hardware. The best
equivalent-cost run (5538.3 GWh, a K=4 seed) sits only 2.4 GWh (0.04%) below
the 5540.7 GWh baseline that spends ~375x more compute, and the K=8..32
plateau beats the schedule-mode seed at the same budget (~5530 vs 5529.2 GWh)
while being feasible in 92/100 runs overall (100% for K>=2) where the seed
schedule violates the boundary. K=8 remains the reference config: the
nominal K=24 winner's +0.25 GWh edge is within noise, and K=8's held-out
ROWP result stands: 10/10 feasible, mean 4228.5 GWh, 0.43% below its
500-multistart baseline.
