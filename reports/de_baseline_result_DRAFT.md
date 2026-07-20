# DRAFT — baseline-seeded DE result (NEEDS HUMAN APPROVAL)

**Not merged into README or paper.** This reframes the headline; do not
publish without approval. Numbers below are read from repo artifacts, not
from the task brief.

## What the commit "Baseline-seeded DE confirmed: ROWP=4271.9" refers to

`tools/optimize_bump_family.py` runs `scipy.optimize.differential_evolution`
over a **13-parameter** parameterization of Claude `iter_192`'s schedule
*form* (cosine LR + warmup + two Gaussian bumps + `alpha = C·alpha0·lr_init/lr`
+ quadratic late boost + fixed betas). The family was designed **post-hoc**
from the LLM's outputs.

The `--init baseline` condition seeds DE at a **neutral zero-bump TopFarm-style
point** (A1=A2=0, no warmup, betas 0.1/0.2 — no LLM-specific values).

## Numbers (exact, from artifacts)

| quantity | value | source |
|---|---|---|
| deployed `iter_192` held-out ROWP | **4271.49 GWh** (feasible; max feasible ROWP over 423 attempts) | `runs/schedule_only_5hr/attempt_log.json` (attempt 192) |
| baseline-seeded DE best held-out ROWP | **4271.85 GWh** (feasible; eval 232) | `runs/archive/bump_opt_baseline/bump_opt_log.json` |
| Claude-seeded DE best held-out ROWP | **4272.85 GWh** (eval 111) | `archive/local/results_bump_opt/bump_opt_log.json` |
| DE − deployed | **+0.36 GWh (0.008%)** | — |

DE setup: `differential_evolution`, seed 42, `polish=False`, **316 evaluations**;
objective = training AEP (−100 infeasibility penalty); held-out ROWP backfilled
per eval and the reported figure is the **max feasible ROWP** — the same
post-hoc "best feasible held-out" selection rule used to pick `iter_192`, so
the comparison is apples-to-apples.

## Draft text (choose one; for Results or Limitations)

> Classical differential evolution, run inside the same dual-Gaussian-bump
> schedule family and seeded from a neutral zero-bump TopFarm-style point,
> reaches 4271.9 GWh on the held-out farm under the identical post-hoc
> best-feasible selection — matching the deployed schedule's 4271.5 GWh to
> within seed noise (±0.4 GWh; one Claude schedule spans ~8 GWh across init
> seeds). The durable contribution is therefore the *parameterization* the
> agent discovers: once its schedule form is fixed, classical tuning finds an
> equally good point inside it. The specific bump placement is not
> load-bearing.

## Three honesty caveats (must be stated if used)

1. **"Matches", not "beats"** — +0.36 GWh is inside seed noise.
2. **Both numbers are post-hoc held-out selections**, not blind test scores
   (the benchmark allows this; the rule is symmetric). DE's best *training*
   solution generalizes to only ROWP 4262.51, so 4271.85 depends on the same
   held-out peeking as the deployed number.
3. **"No LLM hint" applies to the seed point, not the search space** — DE
   searches inside the LLM-discovered family. This does *not* show "plain DE
   matches the LLM"; the search space is the LLM's contribution.
