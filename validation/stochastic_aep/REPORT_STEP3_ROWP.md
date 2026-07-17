# Step 3 (ROWP) — confound + generalization on the IEA 740-10 ROWP

**Verdict: Outcome 1 — disabled-safeguard hypothesis supported.** The
properly-equipped (decaying-LR + ES) baseline reaches 20/20 strict
feasibility on the 740-10. The Part 3 "iter_192 wins on feasibility"
result was contingent on a denuded baseline (constant LR, ES couldn't
fire). The honest paper claim shrinks to the substitution result.

## Setup

- Farm: IEA 740-10 ROWP irregular, N = 74, IEA 10 MW (D = 198 m),
  Borssele 6-vertex polygon, NOJ k = 0.05 + SquaredSum (Part 1
  engine-validated).
- Min spacing: 2 D = 396 m.
- Wind: 12-sector Weibull(A, k) from the official Wind_Resource.yaml
  (same as Part 1).
- Stochastic AEP: K = 50 MC draws per gradient call (Part 2 unbiased,
  z = −0.58 on this farm). Operational envelope clamp [4, 25] m/s.
- 8000 SGD iters per restart, 20 restarts per cell, shared
  `init_seed = r`, `sample_seed = r + 200_000` across cells.
- ES threshold = 0.1.

Plumbing fixes specific to ROWP:
1. UTM coords (~5 × 10⁶ scale) translated to origin-centred local
   coords before SGD (constraint penalty ~ distance² → UTM scale
   inflates bp by ~ 10¹⁰× and destabilises Adam).
2. Borssele polygon ships CW; pixwake's `boundary_penalty` /
   `wind_aware_init` assume CCW. Reversed vertices for SGD.
3. Both fixes are no-ops on AEP (translation-invariant) and on
   pixwake↔pywake equivalence (Part 1 stands).

## Pre-flight gate (passed)

`topfarm_default_decay` schedule: lr starts at 50, bisection-decays to
gamma_min = 0.5 over 8000 steps. lr_ratio trajectory:

| iter | lr_ratio |
|---:|---:|
| 0 | 1.000 |
| 2000 (25 %) | 0.750 |
| 4000 (50 %) | 0.316 |
| 6000 (75 %) | 0.075 |
| 8000 (100 %) | 0.010 |

`lr_ratio ≤ 0.1` **first crosses at iter 5656** (70.7 % through). ES
fires in all 20 restarts of the ES-ON cell. **The mechanism actually
worked this time** — not the DEI no-op.

## Cell results

| cell | strict feas /20 | practical feas /20 | bp range | spacing penalty range | min pair dist (m) | AEP mean ± std (GWh) | best AEP (GWh) |
|---|---:|---:|---|---|---|---|---:|
| `topfarm_default_decay` **OFF** | 0 | 0 | [9.7, 172] | [19, 1134] | [395.2, 396.0] | 3259.02 ± 0.77 | 3260.72 |
| `topfarm_default_decay` **ON** (pivotal) | **20** | **20** | [0, 0] | [0, 0] | [396.0, 397.0] | 3259.03 ± 0.96 | 3261.14 |
| `funwake_iter192` **OFF** | 0 | 19 | [2 × 10⁻⁴, 2 × 10⁻³] | [0, 2.4] | [396.0, 396.1] | 3261.40 ± 1.05 | 3263.07 |
| `funwake_iter192` **ON** | **20** | 20 | [0, 0] | [0, 0] | [397.0, 402.8] | 3261.18 ± 1.39 | 3264.24 |

ES trigger info:

| cell | trigger fraction | iter first crossing |
|---|---:|---:|
| `topfarm_default_decay` ON | **20 / 20 (100 %)** | 5656 (70.7 %) |
| `funwake_iter192` ON | 20 / 20 (100 %) | 0 (warmup phase) |

(`iter_192`'s schedule has a warmup ramp-up that starts at `lr = 0`,
crossing the ES threshold at iter 0. This is a quirk of measuring
`lr_i / lr_0` on a non-monotonic schedule. The substantive ES firing
happens later, during the cosine decay phase. It does not affect the
pivotal-cell outcome.)

## Interpretation per the user's rules

### Pivotal cell (decay-baseline + ES) reached feasibility 20/20.

This is **Outcome 1 — disabled-safeguard hypothesis is supported**.

The Part 3 / DEI Step 3 "iter_192 reaches feasibility 20/20 vs
sgd_baseline 0/20" finding is **partly an artifact of a denuded
baseline**: the constant-LR `sgd_baseline` could not engage ES (its
lr_ratio stays at 1.0 forever; ES never fires) and was not given the
α-escalation safeguard that iter_192 includes by construction. When the
baseline is properly equipped — TopFarm-default decaying LR + Quick
2023 Algorithm 1 early-stopping — it reaches feasibility at the same
rate as iter_192. The original feasibility win was a comparison against
a baseline missing its standard tools.

### Substitution confirmed: same destination, different routes.

The honest claim:
- **iter_192 reaches feasibility via penalty-escalation** (its α schedule
  grows to ~ 5 α₀ × lr_peak / lr_t + α₀ × t² late boost). Reaches
  practical feasibility 19/20 even WITHOUT ES.
- **Decay-baseline reaches feasibility via early-stopping** (Quick 2023
  Algorithm 1: drop AEP gradient when lr_i / lr_0 ≤ 0.1; let
  constraint gradient drive the final 30 % of training).
- Both reach 20/20 strict feasibility on ROWP when their respective
  mechanism is engaged. iter_192 + ES is redundant; baseline alone is
  insufficient.

The LLM **re-derived the feasibility mechanism by a different
mathematical route** (gain scheduling on the penalty multiplier vs
gradient switching). That is a substantive — and interesting — finding
for the paper. It is **not** "iter_192 beats the baseline on
feasibility." The Step 3 (DEI) report already framed this as Outcome 4
(substitution); the ROWP results now confirm that the substitution
holds on a held-out farm, not just on the training farm.

### AEP differences are small and within MC noise.

- Pivotal cell vs iter_192 + ES: ΔAEP = 3261.18 − 3259.03 = **+ 2.15 GWh
  (+ 0.066 %)** in iter_192's favour. With σ ≈ 1.0 GWh per restart, the
  paired SE ≈ 0.3 GWh, so t ≈ 7 — statistically significant but
  practically trivial.
- AEP of iter_192 + ES vs iter_192 + OFF: − 0.22 GWh (within noise);
  the cost of ES strict-feasibility cleanup is negligible on this farm.

The take-away: iter_192 may retain a small AEP edge (~ 0.07 % on ROWP,
similar to DEI), but the **feasibility advantage attributed to it in
the original FunWake comparison evaporates** under a properly-equipped
baseline.

## What this kills

The published / draft FunWake claim — that the LLM-evolved iter_192
schedule outperforms the standard baseline on feasibility — must be
withdrawn or substantially reframed. The honest replacement claim is:

> *We tested whether an LLM-evolved schedule could replace a known
> feasibility-enforcement mechanism (early-stopping per Quick 2023
> Algorithm 1) by a different mathematical route. Across both training
> (DEI) and held-out (740-10) farms, the LLM-evolved iter_192
> reaches feasibility through penalty escalation, with feasibility
> outcomes matching a TopFarm-default decaying-LR baseline with ES
> enabled. AEP differences are within MC noise. The result is a
> substitution finding — the LLM rediscovered a known feasibility
> pattern via gain scheduling — not a strict superiority finding.*

## Caveats

1. **The lr-decay off-by-one between pixwake and TopFarm (Step 2)
   cancels here.** All four ROWP cells run pixwake's SGD with pixwake's
   bisection convention. The relative comparison among cells is
   internally consistent; do NOT compare these AEP numbers to the
   published 3429.63 — the Step-2 off-by-one and the SGD vs deterministic
   AEP eval differ uncancelled across engines.
2. **`iter_192` ES trigger at iter 0** is a measurement quirk from the
   schedule's warmup phase, not an implementation bug. The substantive
   late-stage ES firing on iter_192 happens after warmup completes.
   Outcome unaffected: iter_192 already reaches 19/20 practical
   feasibility without any ES, so ES on it is mostly cosmetic strict-
   tightening.
3. **N = 20 per cell**, K = 50 MC draws per gradient. Increasing either
   would tighten the AEP error bars but is unlikely to change the
   feasibility outcome (20/20 vs 0/20 is categorical).
4. **No tuning of threshold, lr, or restart count was done to chase
   any outcome.** Pre-flight check confirmed the schedule actually fires
   ES; pivotal cell ran as configured; result reported as observed.

## Artifacts

- `validation/stochastic_aep/run_step3_rowp.py` — driver with CCW +
  translation fixes and pre-flight gate.
- `validation/stochastic_aep/step3_rowp.json` — all 80 restart records,
  per-cell summaries, pre-flight, trigger info.
- `validation/stochastic_aep/step3_rowp.log` — stdout trace.
- `validation/stochastic_aep/problem_740.json` and
  `rowp_weibull_12.json` — 740-10 problem & resource extracted from
  official yaml in Part 2.
- ~64.5 min wall on local CPU (3869 s for 80 restarts).
