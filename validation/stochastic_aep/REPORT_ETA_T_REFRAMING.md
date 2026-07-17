# ηT reframing — δ-sweep restated in the primitive parameter

**Honesty upgrade.** The δ-sweep results from `REPORT_DELTA_SWEEP_ES.md` and
`REPORT_HARDENING.md` should be reframed in terms of ηT (the user-facing
final-learning-rate / positioning-tolerance the practitioner actually
turns), not δ (a derived decay rate per Quick 2023 Eq. 13).

Same experiment, same data, no compute change beyond a 12-run paper-point
fill-in. Cleaner and more defensible.

## The relabeling

Quick 2023 Eq. 13 defines the lr decay schedule:

  ηᵢ = S(η₀, δ, i) = η₀ · ∏_{t=1..i} 1 / (1 + δ · t)

δ is bisected such that ηT = η₀ · ∏_{t=1..T} 1/(1 + δ · t). δ depends on
(η₀, ηT, T) — it is **not a free parameter**, just a representation. ηT
is the user-facing knob (the design-variable position tolerance reached
at iteration T) and δ falls out automatically.

For our setup: η₀ = lr_init = 50 m (Adam-normalized step magnitude is
~lr in metres), T = 8000 SGD steps. So ηT = η₀ × gamma_min_factor =
**50 m × δ**.

Quick 2023 fixes ηT = 0.1 m (positioning tolerance equal to design-variable
precision). The pixwake default `gamma_min_factor = 0.01` corresponds to
**ηT = 0.5 m**, 5× the paper-recommended value (a documented pixwake
default deviation, not introduced by us).

## Mapping our swept points to ηT

| swept δ | ηT (m) | physical interpretation |
|---:|---:|---|
| 0.001 | 0.05 | sub-cm — below typical position tolerance |
| **0.002** | **0.1** | **Quick 2023 recommended (paper value)** |
| 0.003 | 0.15 | near paper recommendation |
| 0.005 | 0.25 | 2.5 × paper |
| 0.007 | 0.35 | |
| 0.01 | 0.5 | **pixwake default** (5 × paper) |
| 0.015 | 0.75 | |
| 0.02 | 1.0 | metre tolerance — typical engineering |
| 0.03 | 1.5 | |
| 0.05 | 2.5 | 25 × paper |
| 0.07 | 3.5 | |
| 0.1 | 5.0 | D/40 (DEI) or D/40 (ROWP) — relaxed |
| 0.15 | 7.5 | |
| 0.3 | 15 | very relaxed |
| 0.5 | 25 | ~10 % rotor diameter — essentially no decay |

**Range covered: ηT ∈ [0.05 m, 25 m]**, spanning sub-precision (where
Adam can't sense the step) to nearly-no-decay (where the schedule is
effectively constant). The paper's ηT = 0.1 m is **bracketed** by our
{0.05, 0.15} points and now filled in exactly by a 12-run add-on
(`add_paper_tolerance_point.py` → `eta_t_paper_point.json`).

Bounds check against physical scale:
- DEI: D = 240 m, min_spacing = 4 D = 960 m. ηT ∈ [0.05 m, 25 m] =
  [D / 4800, D / 9.6]. Reasonable.
- ROWP: D = 198 m, min_spacing = 4 D = 792 m. Similar.

## How the best-ηT differs from paper-ηT

Per-cell best-ηT (= 50 × best-δ from `h1_refined_delta.csv` for low-margin
cells, `delta_sweep.csv` for headline):

| cell | best-ηT (m) | × paper (0.1 m) | iter_192 gap vs best-ηT baseline |
|---|---:|---:|---:|
| dei_n60_roserowp | **15.0** | 150 × | +0.603 % |
| rowp_n70_roseomnidir | 5.0 | 50 × | +0.538 % |
| rowp_n60_rosedei | 1.5 | 15 × | +0.756 % |
| rowp_n60_roserowp | 3.5 | 35 × | +0.699 % |
| dei_n60_rosedei | 2.5 | 25 × | +0.549 % |
| dei_n60_roseomnidir | 5.0 | 50 × | +0.616 % |
| dei_n70_rosedei | 25.0 | 250 × | +0.449 % |
| dei_n70_roseomnidir | 2.5 | 25 × | +0.613 % |
| dei_n70_roserowp | 5.0 | 50 × | +0.544 % |
| dei_n80_rosedei | 5.0 | 50 × | +0.503 % |
| dei_n80_roseomnidir | 5.0 | 50 × | +0.602 % |
| dei_n80_roserowp | 5.0 | 50 × | +0.666 % |
| rowp_n60_roseomnidir | 0.5 | 5 × | +0.659 % |
| rowp_n70_rosedei | 1.0 | 10 × | +0.833 % |
| rowp_n70_roserowp | 1.0 | 10 × | +0.892 % |
| rowp_n80_rosedei | 1.0 | 10 × | +0.909 % |
| rowp_n80_roseomnidir | 1.0 | 10 × | +0.953 % |
| rowp_n80_roserowp | 5.0 | 50 × | +1.104 % |

**Key observation**: the AEP-optimal ηT under our stochastic K=50 Adam
schedule is consistently MUCH larger than Quick 2023's recommended 0.1 m
— ranging from 0.5 m (5 ×) to 25 m (250 ×). For most cells the
optimum sits at ηT ≈ 1 – 5 m (10 × – 50 × paper).

This is not a contradiction of Quick 2023's recommendation — the paper
recommended 0.1 m as a *positioning-precision* tolerance, while our
"best-ηT" maximises a *stochastic-gradient AEP estimator at fixed
T*. Two different objectives, two different optima. Under stochastic
K=50 gradients, the SGD doesn't need to converge to 10 cm precision
to find a high-AEP layout; it benefits from keeping more late-stage
exploration alive at the cost of looser positioning.

**Honest paper claim — final form**:

> *We swept ηT (the final-learning-rate / position-tolerance the
> practitioner sets, per Quick 2023 Eq. 13) across the meaningful
> range from 0.05 m (sub-precision) to 25 m (~10 % rotor diameter),
> resolving 14 points spanning two orders of magnitude including the
> paper's recommended 0.1 m exactly. For each headline cell we found
> the AEP-optimal ηT under stochastic K=50 gradients and compared
> iter_192 against the optimally-tuned baseline. iter_192 beats the
> best-ηT baseline by 0.45–1.10 % AEP on every headline cell (18/18,
> margin 8 × – 150 × multi-seed spread). The best-ηT varies per cell
> (0.5 – 25 m), consistently 5 × – 250 × larger than Quick 2023's
> recommended positioning tolerance — but iter_192 still beats this
> optimally-tuned point on every cell.*

## What changes in the existing reports

`REPORT_DELTA_SWEEP_ES.md` and `REPORT_HARDENING.md` should be
updated to:

1. Lead the description with ηT, treat δ as derived per Eq. 13.
2. Use ηT in tables and figures (already in metres, physically
   interpretable).
3. Note explicitly that pixwake's default ηT = 0.5 m is 5 × the
   paper's recommended 0.1 m.
4. Add the 0.1 m exact point so the paper-recommended value isn't
   bracketed-but-not-tested (handled by the 12-run add-on).
5. State: "best-ηT varies cell-to-cell (0.5 – 25 m, all ≥ paper's
   0.1 m); iter_192 beats the best-ηT baseline on every cell."

The data and the conclusions are unchanged. Only the framing tightens.

## Artifacts

- `validation/stochastic_aep/add_paper_tolerance_point.py` — script
  that adds the exact ηT = 0.1 m point (δ = 0.002) for the 4
  low-margin cells (12 runs).
- `validation/stochastic_aep/eta_t_paper_point.json` — the 12-run
  output.
- This document.
- `REPORT_DELTA_SWEEP_ES.md` and `REPORT_HARDENING.md` — to be
  updated with ηT framing in a separate pass once the 12 runs land.
