# Fair-baseline 48-cell matrix re-eval (decay + ES SGD, 3 seeds)

**Verdict: Outcome 1 — gap survives against a properly-equipped standard
baseline.** Even when the baseline is the published Quick 2023 SGD
(decaying LR + Algorithm 1 early-stopping; the safeguard the
constant-LR `sgd_baseline` was missing), the LLM-evolved schedules
maintain a 0.2–1.1 % AEP advantage on multidirectional roses with
N ≥ 40, on both polygons, with multi-seed spread of ~ 0.01–0.06 %.

This is a stronger result than the prior matrix re-eval against the
500-multistart `topfarm_sgd_solve` denominator. The Step-3 "feasibility
artifact" framing was correct for that confound; for the deploy-AEP
question, the LLM-evolved schedules carry a real scoped advantage even
against the standard fair opponent. Substitution result holds AND a
scoped superiority result holds on top.

## Setup

- 48 cells (2 polygons × 4 roses × 6 N values = 30, 40, …, 80).
- BastankhahGaussianDeficit(k = 0.04), 4 D min spacing, same matrix
  fixture as `run_matrix_stochastic.py`.
- Stochastic K = 50 categorical-rose sampling per gradient call.
- Adam SGD via the run_step3 skeleton, 8000 iters, single init seed
  (= 0). Matched compute across all three schedules.
- **Three** sample seeds per (cell, schedule): 100 000, 200 000,
  300 000. Per-cell spread is empirical, not borrowed.
- CCW polygon fix + UTM→local translation applied per cell. Init bp
  = 0 across all 432 runs.

Schedules:
- **decay_es_baseline** (PRIMARY FAIR baseline): TopFarm-default LR
  decay from lr_init = 50 to lr_init × 0.01 over 8000 steps, β₁ = 0.1,
  β₂ = 0.2, Quick 2023 Algorithm 1 early-stopping with threshold 0.1.
  (`run_step3_rowp.topfarm_default_decay` + ES = ON.)
- **claude_iter192**: Claude's deployed schedule, ES = OFF (its
  α-escalation handles feasibility per Step 3).
- **gemini_iter192**: Gemini's deployed schedule (8-cycle cosine
  restarts, cyclic betas, late squeeze), ES = OFF.

ES pre-flight: decay_es_baseline lr_ratio first crosses threshold at
iter 5656 / 8000 (70.7 %). All 144 baseline runs trigger ES (100 %
trigger fraction). bp_final = 0 across 144/144 baseline runs —
feasibility reached.

## Headline results

| | Claude | Gemini |
|---|---:|---:|
| Cells clearing FAIR bar (gap ≥ 0.2 % AND gap > seed-spread) | **38/48** | **35/48** |
| Cells clearing in uniform rose (12 cells) | 2/12 (DEI N=70, 80) | 0/12 |
| Cells clearing in multidir roses (36 cells) | 36/36 | 35/36 |
| Largest gap | 1.12 % (ROWP × ROWP-rose × N=80) | 0.89 % (DEI × omnidir × N=80) |

Per-cell spread (multi-seed std / baseline mean) is **0.01 – 0.06 %**
typically — gaps that clear 0.2 % do so by a 5 – 20 × margin in
practice.

## Pattern (figure: matrix_fair.{pdf,png})

- **Uniform rose**: small gaps. Mostly null, but DEI polygon × N=70/80
  nudges above 0.2 % (still tiny, spread-tight). ROWP × uniform × N ≤ 50
  is exactly 0 (turbines spread along the single-direction crosswind
  axis identically for all schedules). The pre-registered "uniform null"
  is **not** absolute against a single-restart baseline — but ROWP
  polygon does hold the null nicely.
- **Omnidirectional**: gaps 0.3 % → 1.0 % monotonic in N. Clears bar at
  N=30 already.
- **DEI rose**: similar to omnidir, gaps 0.3 % → 0.9 %.
- **ROWP rose**: largest, gaps 0.3 % → 1.1 %. **ROWP polygon × ROWP
  rose × N=80** is the peak cell for Claude.

The **held-out polygon (ROWP) shows LARGER gaps than the training
polygon (DEI)** at every multidirectional rose × N pairing. The
schedule advantage **generalizes** and even amplifies on held-out.

Claude consistently > Gemini by ~ 0.1–0.2 % across cells; ranking is
stable.

## Comparison to previous matrix re-evals

| Reference  | Claude clears 0.2 % | Setting |
|---|---:|---|
| Original coarse-grid (vs 500-multistart baseline) | 24/48 | Old matrix |
| Stochastic re-eval (vs 500-multistart baseline) | 26/48 | Last run |
| **Fair-baseline (vs decay+ES single-restart, K=50, 3 seeds)** | **38/48** | This run |

Why more cells clear vs decay+ES than vs 500-multistart: the decay+ES
baseline is **single-restart**, like Claude/Gemini are. The
500-multistart baseline gets the "best of 500 attempts" advantage that
single runs don't. So:

- vs 500-multistart: Claude/Gemini schedules beat best-of-500-default
  multistart at 26/48 cells.
- vs single decay+ES: Claude/Gemini schedules beat single-restart
  decay+ES at 38/48 cells.

Both comparisons are honest; pick the right one for the paper claim.
The fair single-restart comparison is more reproducible for users
without 500x compute.

## Interpretation per pre-registered rules

This is **Outcome 1 — gap survives against decay+ES SGD**. Honest
paper claim:

> *Across 48 matrix cells, LLM-evolved schedules (Claude, Gemini) beat
> a properly-equipped standard baseline (Quick 2023 Algorithm 1 SGD:
> decaying LR + early-stopping) by 0.2 – 1.1 % AEP at matched compute,
> on multidirectional wind roses with N ≥ 30. The advantage is robust
> to gradient stochasticity (stochastic K = 50 MC sampling), holds on a
> held-out polygon (in fact amplifies there), and is statistically
> well-separated from per-seed noise (gap > 5 – 20 × spread). On the
> uniform single-direction rose, the gap is small or null — consistent
> with the single-direction layout being too simple to differentiate
> schedules. Result is bounded: scope = multidirectional roses, N ≥ 30,
> both polygons.*

## Caveats and what this does NOT support

1. **Smart-start follow-up not yet run.** User's spec called for
   headline cells vs smart-start init as a secondary fair baseline.
   That's the next run; will refine whether the gap holds against a
   smarter init, not just a smarter schedule.
2. **SLSQP not run.** Optional in spec; deferred.
3. **Single init seed (0) per cell.** Multi-seed is on sampling, not
   on init. Multi-init multistart would shift baseline values
   somewhat; expected to favour baseline slightly (multiple starts
   beat single start on harder cells).
4. **Per-cell wall-time is similar for all 3 schedules** because all use
   the same Adam skeleton and same 8000 iters. No compute confound at
   the cell level — but compare to 500-multistart is NOT compute-
   matched (multistart used 500 × 6000 iters vs single × 8000).
5. **Uniform rose is not a hard null.** DEI × uniform × N=70/80 nudges
   above 0.2 % for Claude. Either decay+ES baseline has slight
   sub-optimality the schedules exploit even on single-dir farms, or
   schedule-driven trajectory diversity provides a small edge at
   high N. Worth noting in paper.
6. **Wake model is BastankhahGaussianDeficit(k=0.04) for all cells**
   (hard-coded in `playground/harness.py`). Other models not tested.
7. **Matrix ROWP polygon (4-vertex)** is a stand-in, not the
   6-vertex 740-10 Borssele polygon. Comparison to published
   3429.63 GWh is NOT valid here — internal-only.

## Artifacts

- `validation/stochastic_aep/run_matrix_fair.py` — parallel driver
  (resume-safe, ProcessPoolExecutor; ran on 1 worker after the 4-worker
  variant was killed for memory).
- `validation/stochastic_aep/matrix_fair.json` — all 432 runs:
  per-seed AEP, bp_init, bp_final, min pair distance, elapsed, ES
  trigger info.
- `validation/stochastic_aep/matrix_fair.log` — stdout trace (both
  attempts concatenated).
- `validation/stochastic_aep/matrix_fair.csv` — per-cell table:
  mean ± std AEP per schedule, gap %, spread %, clears-fair-bar flag.
- `validation/stochastic_aep/matrix_fair.{pdf,png}` — figure with
  error bars (multi-seed spread) per cell.
- `validation/stochastic_aep/analyze_matrix_fair.py` — analysis
  generator.
- ~ 4.9 hr wall on local CPU for the 244 resumed runs (5 hr total
  including 188 from the killed 4-worker attempt).
