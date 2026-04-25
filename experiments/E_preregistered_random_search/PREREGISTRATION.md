# Pre-registration — Random search ablation, version 2

**Date authored:** 2026-04-24 (week before submission deadline)
**Status:** LOCKED. Do not modify after the corresponding `family.py`
is committed and the search has been launched. If the family needs to
change after seeing results, create `family_v3.py` and a new
PREREGISTRATION_v3.md, and disclose the change in the paper.

## Why a v2 is needed

The original random-search ablation (`tools/random_search_ablation.py`)
was designed AFTER inspecting the schedules the LLMs produced. Its
parameter family includes Gaussian bumps and sinusoidal shake — the
exact structures the LLMs proposed. The paper currently flags this
post-hoc design honestly, but a reviewer can argue (correctly) that
the ablation tests "are LLM-found structures findable by random
search?" rather than "would random search have found them
independently?"

This v2 ablation answers the second question: design a family
*before* inspecting LLM outputs, then run random search inside it,
then compare to the LLM scores.

## Design constraints (the only thing we are allowed to look at)

1. **Inputs**: the same 4-output schedule signature
   `schedule_fn(step, total_steps, lr0, alpha0) -> (lr, alpha,
   beta1, beta2)`.
2. **Compute envelope**: same 8000-step Adam skeleton.
3. **Generic parameterization**: each output is a smooth function of
   normalized progress `t = step / total_steps`, expressed as a small
   Fourier basis. No hand-coded "bump", "restart", "squeeze", or any
   structure the LLMs proposed.

## Parameter family (locked)

For each of `lr`, `alpha`, `beta1`, `beta2`, the schedule is

    f(t) = base * exp(c0 + sum_{k=1..K} a_k cos(2 pi k t) + b_k sin(2 pi k t))

with `K = 4` Fourier components per output. Free parameters per output:
1 (c0) + 4 (a_k) + 4 (b_k) = 9. Total free parameters across 4 outputs
= 36.

Bases (kept fixed; chosen to make the family include simple constant
schedules and a single cosine cycle as low-Fourier members):

    base_lr     = lr0
    base_alpha  = alpha0
    base_beta1  = 0.9
    base_beta2  = 0.999

Sampling distribution per coefficient: `c0 ~ N(0, 0.5)`,
`a_k, b_k ~ N(0, 0.5/k)` (decaying, so high-frequency components are
rarer than low-frequency). Truncate `lr`, `alpha` to non-negative
via the `exp` envelope; clip `beta1` to `[0.5, 0.999]` and `beta2`
to `[0.9, 0.9999]` post-evaluation.

## Search budget (locked)

- N samples: 320 (same as the post-hoc family for fairness)
- Random seed: 0 (the SAMPLER seed; the skeleton init seed is 0)
- Both farms: train (DEI, N=50) and ROWP (N=74) for every sample
- Score: AEP at the schedule's terminal step

## Success criterion (locked)

- Best feasible ROWP across 320 samples vs. the LLM-discovered max
  (4271.5 for Claude, 4269.3 for Gemini, single-seed numbers from
  the existing paper).
- If the v2 best ≥ LLM best within 5 GWh: the parameterization itself
  is enough; the LLM contribution narrows further.
- If the v2 best is materially lower (>10 GWh below LLM):
  the LLM's *structural* contribution (e.g., placing energy in
  bump-shaped components rather than spread across all Fourier modes)
  is the real value. This is the more interesting outcome and would
  strengthen the paper.

## Pre-registration honesty pact

We commit to reporting the v2 result regardless of which way it falls.
If v2 beats LLM, the paper's narrative narrows to
"the LLM contribution is parameterization design, easily replaced by
a generic basis". If v2 trails, the LLM's structural prior matters.
Both are publishable; only one is what we currently claim.
