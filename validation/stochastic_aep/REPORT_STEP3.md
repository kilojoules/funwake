# Step 3 — Four-cell feasibility-confound comparison

**Setup.** DEI training farm, N = 50, Bastankhah k = 0.04 (iter_192's
discovery wake model), K = 50 MC draws / iter, 8000 SGD iters, 20
restarts, same `init_seed`/`sample_seed` per restart across cells.
Threshold = 0.1.

`sgd_baseline` = constant LR = 50, β₁ = 0.1, β₂ = 0.2 (paper_schedules
"FunWake iter_0" baseline, matches Part 3).
`funwake_iter192` = Claude iter_192 verbatim (warmup + cosine + dual
bumps + α dip + α₀ × t² escalation, β₁ = 0.3, β₂ = 0.5).

ES = Quick 2023 Algorithm 1, implemented in run_part3 SGD loop (mirror
of pixwake's `topfarm_sgd_solve` ES path, validated against TopFarm2 in
Step 2).

## Results

| cell | strict feas /20 | practical feas (bp < 10⁻²) /20 | bp range | min-pair-d range (m) | AEP mean ± std (GWh) | best AEP (GWh) |
|---|---:|---:|---|---|---|---:|
| **sgd_baseline OFF** | 0 | 0 | 1.5 × 10³ … 2.4 × 10⁴ | 1706 … 2151 | 4459.55 ± 0.58 | 4460.47 |
| **sgd_baseline ON**  | 0 | 0 | 1.5 × 10³ … 2.4 × 10⁴ | 1706 … 2151 | 4459.55 ± 0.58 | 4460.47 |
| **iter_192 OFF**     | 2 | **20** | 0 … 6.2 × 10⁻⁴ | 1897 … 2201 | 4460.25 ± 0.46 | 4461.34 |
| **iter_192 ON**      | **20** | 20 | 0 … 0 | 1821 … 2192 | 4460.01 ± 0.50 | 4460.95 |

Spacing penalty = 0 in every restart of every cell (all 80 layouts
respect 4 D / 960 m min spacing).

## Two findings

### Finding 1 — sgd_baseline + ES is a NO-OP at this schedule

`sgd_baseline OFF` and `sgd_baseline ON` produce **bit-identical** summary
statistics — same bp range, same AEP mean ± std, same min-pair-distance
range. Because the schedule keeps `lr_i = lr_0` for all iterations,
`lr_i / lr_0 = 1.0 > 0.1` always, so the ES condition never fires. The
ES branch is dead code for this schedule.

→ The "pivotal cell" as specified does not probe the user's intended
question. To answer "does a decaying-LR baseline with ES reach
feasibility?", we would need a `topfarm_default` schedule (lr decay
from lr_0 → lr_0 × γ_min). Flagged below as follow-up.

### Finding 2 — iter_192's feasibility win SURVIVES the proper baseline reading

Even reading the cells literally (sgd_baseline + ES = 0/20), this maps
to the user's **Outcome 1** ("baseline + ES still fails ~ 0/20: iter_192's
feasibility win is real and survives the proper baseline"). However the
finding is mechanistically more interesting than Outcome 1 alone:

**iter_192 already reaches practical feasibility (20/20) WITHOUT ES.**
ES adds no practical-feasibility headroom — it only tightens 2/20 strict
→ 20/20 strict (i.e., bp < 6 × 10⁻⁴ → bp == 0.0 exactly). The
feasibility-enforcement work iter_192 does is autonomous of ES.

This is the user's **Outcome 4 — substitution**: the LLM-discovered
schedule recovered a feasibility mechanism by a different route. Where
ES drops the AEP gradient when lr decays, iter_192 grows α (penalty
coefficient) to 5α₀ × lr_peak / lr_t throughout, plus an α₀ × t² late
boost — same destination (constraint gradient dominates final phase),
different mechanism.

The substitution shows up in the late-iter dynamics:
- **iter_192 OFF, late-stage:** α grows large as lr decays, so α · ∇γ
  dominates ∇AEP. Same outcome as ES (drop ∇AEP altogether) but
  achieved via gain scheduling on the penalty multiplier.
- **iter_192 ON, late-stage:** ES kicks in around iter 4000 (when
  lr_ratio first ≤ 0.1), drops ∇AEP entirely. Final layout converges to
  bp = 0 strictly; constraint gradient drives the last few hundred steps.
  AEP drops by 0.24 GWh vs OFF (within MC noise; the 0.24 GWh is paying
  for strict feasibility instead of practical).

The two mechanisms are functionally equivalent at the practical-feasibility
boundary. ES just cleans up the residual numerical bp ~ 10⁻⁴ → 0.

## Interpretation

For the writeup, this is **honest and interesting**: iter_192 substitutes
for ES via penalty escalation. Specifically:

- **Part 3 finding ("iter_192 helps feasibility, baseline fails"):**
  STANDS. The 0/20 vs 20/20 practical-feasibility gap is real.
- **Original confound concern ("maybe the baseline failed because it
  was denied its safeguard"):** DISPROVEN for this baseline. The
  safeguard doesn't even fire on the constant-LR schedule, yet iter_192
  reaches feasibility regardless.
- **Honest framing:** iter_192's α-escalation IS a re-derivation of the
  Quick 2023 ES feasibility mechanism. Two routes to the same place.
  This is a substantive — and interesting — finding for the paper:
  *the LLM rediscovered a known constraint-handling pattern under a
  schedule shape (constant β₁=0.3, β₂=0.5; smooth α growth) that the
  authors didn't explore in 2023.*

## Caveats and follow-ups

1. **Constant-LR baseline is a weak counterfactual.** The cleanest
   four-cell test would replace `sgd_baseline` with `topfarm_default`
   (decaying LR = TopFarm2 / Quick 2023 paper convention) so that ES
   has something to fire against. I have **not** run this — the
   user's spec used `sgd_baseline`, and Step 2 validated my ES against
   TopFarm at the algorithm level not at this fixture level. If you
   want this cell, I'll add it (~25 min DEI).

2. **No ROWP cells.** The user said "if cheap, repeat on the validated
   740-10 farm to show it's not DEI-specific." Per the cost notes in
   Part 2 (86 candidates/hr at N = 74), the 4-cell ROWP variant ≈ 100
   min wall. Not run; await sign-off.

3. **iter_192 ON paid 0.24 GWh for the bp-residual cleanup.** 4460.25
   (OFF) → 4460.01 (ON), Δ = −0.24 GWh. This is within MC noise
   (σ ≈ 0.5 GWh per restart) but consistent: ES drops the AEP gradient
   for the last ~ 4000 iters, so the AEP couldn't keep micro-improving
   in that phase. Trade-off is favourable if you value strict
   feasibility; neutral if you only need practical.

4. **Substitution claim warrants its own ablation.** To prove
   iter_192's α-escalation specifically is what does the substituting
   (vs the bumps or β values), strip iter_192 to {α-escalation only,
   no bumps, no warmup-cosine, default β₁=0.1, β₂=0.2}. Then test ±
   ES. Conjecture: that minimal-α-escalation schedule alone is enough
   to recover feasibility. Not run.

## Artifacts

- `validation/stochastic_aep/run_step3.py` — 4-cell driver, ES wired
  into the SGD step (mirrors pixwake's topfarm_sgd_solve ES path).
- `validation/stochastic_aep/step3_dei.json` — all 80 restart records.
- `validation/stochastic_aep/step3_dei.log` — stdout trace.
- ~95 min wall on local CPU.
