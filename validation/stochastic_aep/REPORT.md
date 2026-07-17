# Part 2 — Stochastic AEP gradients restored (Quick-2023 style)

**Purpose:** replace FunWake's mean-speed-per-sector AEP objective with
the K-sample stochastic Weibull-marginalized estimator the parent SGD
paper (Quick et al. 2023, Eqs. 5–6, 16–17) uses, so iter_192 can later
be tested under the objective it was supposed to inherit.

**Status:** PASS. Estimator runs, is unbiased of the truncated Weibull
integral on the operational envelope, fits inside the existing 1-hour
agent budget.

## Implementation

`validation/stochastic_aep/stochastic_aep.py:stochastic_aep_factory()`

Per iteration (K = 50, default):
1. Sample sector index `j ~ Categorical(sector_probability)`.
2. Sample direction offset uniformly within the sector
   `wd ~ Uniform(center_j − w/2, center_j + w/2)`.
3. Sample speed via inverse-CDF Weibull
   `ws = A_j · (−ln(1 − u))^(1/k_j)`, `u ~ Uniform(0, 1)`.
4. Clamp `ws` to the turbine operational envelope `[cut_in, cut_out]`
   (zero-weighted outside; see note below).
5. Call `pixwake.WakeSimulation(...)` on the 50-case batch; AEP_hat =
   `r.aep(probabilities=(1/K)·in_range)`.

**Why clamp.** Pixwake's `Curve` linearly extrapolates a tabular power
curve outside its sampled support. The IEA 740-10 turbine's curve runs
4–25 m/s; samples below 4 (cut-in) extrapolate to ~387 kW and above
25 (cut-out) hold at rated ~10 MW — both physically wrong. Masking
out-of-range samples (zero probability) makes the estimator unbiased of
the same truncated integral the deterministic Riemann sum approximates.
The DEI 15-MW turbine has a curve from 0 m/s upward but still ends at
25, so only the upper clamp activates for DEI.

**No quadrature grid.** The SGD path is K = 50 native draws per outer
iteration. The 360 × 22 = 7920-bin Riemann sum used in the sanity check
below is *not* part of the SGD loop — it is the ground-truth reference
for unbiasedness only. The user's concern about an 8280-bin objective
collapsing the budget does not apply.

## Sanity check — stochastic mean ≈ deterministic ground truth

Repeats: 1000 independent K=50 MC estimates per cell. Wake/spacing held
fixed to iter_192's discovery setup (DEI: Bastankhah k=0.04) and the
published 740-10 setup (NOJ k=0.05) respectively.

| Farm        | N  | Wake               | Det. fine-grid (GWh) | MC mean ± SE (GWh)  | Δ (GWh) | Δ (%)  | z      | verdict |
|-------------|---:|--------------------|---------------------:|---------------------|---------|--------|--------|---------|
| DEI train.  | 50 | Bastankhah k=0.04  | 4401.44              | 4398.53 ± 9.94      | −2.90   | −0.07% | −0.29  | unbiased ✓ |
| 740-10 IRR  | 74 | NOJ k=0.05         | 3413.71              | 3407.08 ± 11.35     | −6.62   | −0.19% | −0.58  | unbiased ✓ |

z-scores |z| ≪ 2 → both consistent with "MC estimator is unbiased of
the truncated Weibull integral." A *single* K=50 draw has σ ≈ 314 GWh
(DEI) / 359 GWh (ROWP) — large; gradient noise but unbiased mean,
exactly what SGD wants.

## Per-iteration cost — fits the 1-hour budget

Measured on local laptop (M2 Pro, CPU JAX, fp64). For each cell, after
JIT warm-up.

| Farm       | N  | Wake               | forward (ms) | forward + grad (ms) | sec / 8000-iter candidate | candidates / hour |
|------------|---:|--------------------|--------------|---------------------|---------------------------|-------------------|
| DEI train. | 50 | Bastankhah k=0.04  | 3.55         | 7.66                | 61.3                      | **~59**           |
| 740-10 IRR | 74 | NOJ k=0.05         | 2.49         | 5.22                | 41.8                      | **~86**           |

Both well above the original 1-hour-budget collapse worry. The N=74
case is faster per-iter than N=50 because NOJ is closed-form-cheap
relative to Bastankhah-Gaussian; the N² wake-pair count is not the
dominant cost at these sizes on CPU.

Original deterministic mean-speed objective (K=12 or 24 cases per
iter) ran at similar throughput in matrix-eval (~30s/candidate per the
schedules_matrix.json times) — so K=50 stochastic is roughly 1.5–2×
deterministic, not the 10–25× pessimistic projection. The hour holds.

## Skeleton integration

Current `playground/skeleton.py:aep_objective(x, y)` has no `key`
argument and is deterministic. To wire the stochastic objective into
the SGD loop:

1. Add `key` to the `step()` carry of `run_loop`.
2. Replace `aep_objective(x, y)` with the stochastic estimator above,
   `aep_objective(x, y, subkey)`.
3. `jax.grad(aep_objective, argnums=(0,1))` differentiates through the
   sampled cases naturally — stop-grad through sampling per JAX
   semantics is automatic for the inverse-CDF/categorical path used
   here (sampling is data, not parameter).

This rewiring is the next subtask before Part 3 deployment. The
factory function returns an `aep_fn` whose signature already takes the
`key` argument; only the skeleton's outer SGD loop needs updating.

## Caveats

1. **Weibull fit for DEI** is derived from the 10-year daily-mean
   time-series at `playground/pixwake/energy_island_10y_daily_av_wind.csv`
   via per-sector MLE (`fit_weibull.py`). This is an
   **internal-paper-equivalent** rose — the parent paper Quick 2023
   uses a binned rose; our fit produces 12-sector Weibull(A, k) with
   A ∈ [9.94, 12.81], k ∈ [2.66, 3.16]. Sector probabilities match
   what the methodology doc describes (SW/W dominant, 0.10–0.14).
   Daily averaging discards diurnal variation — flagged.
2. The deterministic ground-truth Riemann sum uses a sector-uniform
   distribution of fine direction bins. Pywake interpolates A/k
   linearly between sector centers. For 12-sector roses this matters
   little (~0.1 % AEP per Part 1c).
3. Stochastic estimator is unbiased for the AEP forward pass; gradient
   bias (zero-mean noise → unbiased gradient via standard SGD theory)
   was not explicitly tested. Standard for inverse-CDF + categorical
   sampling: reparameterized, so `jax.grad` propagates correctly
   without additional REINFORCE-style estimators.
4. Cost numbers are on CPU; on a single LUMI GPU these will be ~5–10×
   faster, raising candidates/hour to ~300–800. Sets the upper bound
   on Part 3 multistart budget.

## Artifacts

- `validation/stochastic_aep/fit_weibull.py` — Weibull(A,k) per-sector
  MLE fit from time-series CSV.
- `validation/stochastic_aep/build_740_problem.py` — extracts 740-10
  yaml into problem+resource JSON.
- `validation/stochastic_aep/stochastic_aep.py` — unbiased K-sample
  estimator + deterministic ground-truth Riemann sum + cost timer.
- `validation/stochastic_aep/dei_weibull_12.json` — DEI 12-sector
  Weibull resource (incl. 3653 raw samples).
- `validation/stochastic_aep/rowp_weibull_12.json` — 740-10 published
  Weibull resource.
- `validation/stochastic_aep/problem_740.json` — 740-10 problem-JSON
  in funwake's harness format.
- `validation/stochastic_aep/stochastic_aep_dei_clamped.json` — DEI
  result (cell 1 of table).
- `validation/stochastic_aep/stochastic_aep_rowp_clamped.json` —
  ROWP result (cell 2 of table).
