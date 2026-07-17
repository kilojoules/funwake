# Part 3 — confound gate: iter_192 under stochastic gradients

**Question.** iter_192's dual bumps + alpha dip were discovered under a
deterministic mean-speed AEP objective. Are they additive to stochastic
gradients (independent help) or were they substituting for missing gradient
noise (no help under proper stochastic objective)?

**Setup.** DEI training farm, N=50, BastankhahGaussianDeficit(k=0.04) —
identical to iter_192's discovery wake model and spacing (480 m / 2D).
K=50 MC draws per gradient call, 8000 SGD iters per restart, 20 restarts
per schedule, same restart-index → same wind-aware-grid init for both.
Total: 40 candidates, ~50 min wall on CPU.

## Result

### Feasibility (boundary + spacing)

| Schedule        | strict feas (bp = 0 & sp = 0) | practical feas (bp < 10⁻² GWh-pen) | boundary-penalty range            |
|-----------------|-------------------------------:|------------------------------------:|------------------------------------|
| sgd_baseline    | **0 / 20** (0 %)               | 0 / 20 (0 %)                       | **1.48 × 10³ … 2.41 × 10⁴**       |
| funwake_iter192 | **2 / 20** (10 %)              | **20 / 20** (100 %)                | **0 … 6.16 × 10⁻⁴**                |

Spacing penalty was 0 across all 40 restarts (min pair distance 1750–2200 m,
well above the 960 m / 4 D spacing). Constraint dynamics are entirely
boundary-driven.

iter_192 cuts the boundary violation by **7 orders of magnitude** vs
baseline. Practically all 20 iter_192 restarts land inside the polygon
(bp ≤ 6 × 10⁻⁴ — effectively numerical-precision feasible). Baseline
puts every restart kilometers outside the polygon.

### AEP (Weibull-marginalized deterministic eval after training)

| Schedule        | mean AEP (GWh) | std (GWh) | best AEP (GWh) |
|-----------------|---------------:|----------:|---------------:|
| sgd_baseline    | 4453.30        | 0.56      | 4454.18        |
| funwake_iter192 | 4453.98        | 0.45      | 4455.03        |

Paired (same restart index → same init):
- mean Δ (iter_192 − baseline) = **+0.68 GWh**
- SE = 0.16 GWh, **t = 4.32** (n = 20)

The AEP gap is statistically significant but **small in absolute terms
(+0.015 % on a 4453 GWh base)** — well inside the per-eval engine
tolerance for any downstream regret signal and far below the MC noise
floor of a single K=50 estimate (σ ≈ 314 GWh; the multistart averaging
is what extracts the signal).

## Interpretation

iter_192 still helps under stochastic gradients, **but not the way the
deterministic-discovery narrative implied.** The help is concentrated
in **constraint enforcement** (boundary penalty), not AEP optimization:

- **AEP gap:** +0.68 GWh paired = 0.015 % of farm AEP. Real (t = 4.3) but
  practically negligible.
- **Feasibility gap:** infeasible-by-kilometers (baseline) vs
  feasible-to-numerical-precision (iter_192). 7-order-of-magnitude
  improvement in boundary violation.

The structural features of iter_192 that drive this:

1. **Warmup + cosine LR decay** (5 % warmup, then cosine to lr_min =
   lr_peak/10000) — lets the alpha penalty catch up before optimization
   takes big AEP-driven steps.
2. **5 × alpha₀ baseline + 3 × α₀ × t² late escalation** — the alpha
   schedule grows to ~8 × baseline by t=1, multiplying the boundary
   gradient. This is the dominant feasibility driver.
3. **Late α dip at t ≈ 0.6** — relaxes constraint briefly to let AEP
   improve, then re-tightens. Adds a tiny amount of AEP at the cost of
   marginal feasibility risk (consistent with the +0.68 GWh / 1.6 × 10⁻⁴
   bp signal).
4. **Constant β₁ = 0.3, β₂ = 0.5** vs baseline's 0.1 / 0.2 — faster
   first/second moment decay, more responsive to gradient sign changes
   from stochastic noise. Plausible that this matters more under
   stochastic gradients than under deterministic; the previous note
   `project_funwake_rerun_status.md` already attributed Gemini's β
   scheduling to constraint enforcement on the same grounds.

The "dual Gaussian bumps" (0.2 × lr_peak at t = 0.5, 0.3 × lr_peak at
t = 0.75) on the LR curve do not appear to drive either the AEP or the
feasibility difference in this comparison — the late-stage α structure
is what carries the result. Whether the bumps add anything beyond
warmup + decay is not yet isolated; an ablation (iter_192 minus bumps)
would settle it.

## Verdict for the writeup (per the rules in the parent plan)

The result is **not "iter_192 stops helping" but "iter_192 helps differently
than advertised under stochastic SGD":**

- **Feasibility:** iter_192 dominates. Categorical, not a margin call.
  20/20 practically feasible vs 0/20. This is the load-bearing finding
  and matches the prior project memory that β-scheduling acts as
  constraint enforcement.
- **AEP:** statistically present (t = 4.3) but practically null (0.015 %).
  In the discover-on-deterministic / deploy-on-stochastic gap discussion
  this should be reported as "the AEP gain attributed to iter_192's
  bumps + dip in the deterministic objective largely vanishes under the
  stochastic objective; what remains is its α-schedule's constraint
  enforcement."

Honest disclosure for the paper: the schedule was discovered under a
deterministic objective and is being deployed under a stochastic one.
The AEP signal is small enough that we should not over-claim. The
**feasibility win is robust and survives the objective swap.**

## Caveats

1. **N = 20 restarts** per schedule. The pairwise t-statistic is solid
   but the variance estimate is wobbly; t = 4.3 → 99.97% confidence two-
   sided, but a single-cell study isn't generalizable. Replicate on
   ROWP / matrix cells before claiming the AEP signal is real.
2. **8000 iters.** Same as deployed FunWake budget. iter_192's late-α
   schedule may be specifically tuned for this T; sgd_baseline at T = 32000
   might catch up on feasibility (it gets more time at constant LR to
   push out then come back). Not tested here.
3. **Strict-feasibility threshold is arbitrary.** bp = 0.0 is over-strict
   for a JAX-fp64 differentiable penalty — anything ≤ 10⁻³ is on-boundary
   in practice. The practical feasibility check (bp < 10⁻²) is the
   physically meaningful one. Pixwake's existing harness uses bp ≤ 0.0
   strictly which makes iter_192 look 10 % feasible when it's really 100 %
   feasible.
4. **No ablation of which iter_192 feature does the work.** Conjecture:
   the α escalation. The bumps and the α dip both look like they should
   help AEP but their measured contribution is dwarfed by α₀ × t²
   structure. Worth isolating before paper claims about "discovered
   structure."

## Artifacts

- `validation/stochastic_aep/run_part3.py` — stochastic-objective
  skeleton + multistart driver (does NOT touch playground/skeleton.py).
- `validation/stochastic_aep/part3_result.json` — per-restart raw +
  per-schedule summary + paired diff.
- `validation/stochastic_aep/part3_run.log` — stdout from the run
  including every restart's elapsed/AEP/feasibility.
- 2953 s wall (~50 min) on local CPU, 40 candidates (20 × 2 schedules).
