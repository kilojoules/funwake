# Is AEP capped/clipped at 5600 GWh? — investigation

**Task 6 (diagnosis only).** No results, figures, or logs were altered.

## Verdict: there is NO cap at 5600.

Nothing in the scorer, skeleton, logging, or plotting caps, clips, floors,
or ceilings AEP at or near 5600 GWh.

- **Decisive counter-evidence:** the *same scorer* records train AEP from
  ~2000 up to **8938.93 GWh** in `runs/schedule_only_5hr/attempt_log.json`
  (28 in-budget attempts exceed 5600). A hard cap at 5600 is impossible
  given those values exist.
- **Only numeric transform is 2-decimal rounding.** `tools/run_optimizer.py`
  computes raw `aep` at line 218 and the only transforms are `round(aep, 2)`
  at lines 254 and 263. Two-decimal rounding merges values only within
  ±0.005 GWh — it cannot collapse distinct results to 5600.
- **No clip in the skeleton.** `playground/skeleton.py` has no clip/cap/round
  on AEP; its only `jnp.where` (lines 131–132) zeroes the *gradient* for
  optional early-stopping (off by default), not the AEP value.
- **No clip in the benchmark scorer.** `benchmarks/dei_layout.py` `score()`
  returns raw `-float(objective(...))` (lines 242–244, 396–397).
- **Plotting does not clamp.** `paper/make_fig_short_convergence.py` plots raw
  values; the train row's `ylim=(5450, 5615)` (line 108) is only an axis
  window — points outside are counted (`n_below`, line 129), never clamped.

## What the 5600 pile-up actually is

The cluster is at **5599.97** (rounds to 5600.0), appearing in **9 attempts**
(262, 264–271), **all `train_feasible = False`**. It is a genuine physical
attractor, not a code artifact: those schedule variants drive turbines out of
bounds / under minimum spacing, which removes wakes and pushes AEP toward the
wake-free asymptote — so they land at the same degenerate *infeasible* layout.

- **Zero** attempts record exactly `5600.0` in any log
  (`grep '"train_aep": 5600'` returns nothing across all logs).
- The literal token `5600.0` appears only in one schedule's docstring
  (`runs/archive/schedule_5hr/iter_035.py:6`, the agent's own rounded note
  "infeasible best was 5600.0 GWh").

## Implication for a "+59.3 on train" claim

**Not safe as a method improvement.** `5599.97 − 5540.7 = 59.27 ≈ 59.3` traces
directly to the infeasible 5599.97 cluster (all `train_feasible = False`).
Crediting a constraint-violating, wake-free layout's AEP to the method is not
defensible.

Honest feasible train figures over the 5540.7 baseline:

| schedule | train AEP (GWh) | feasible | gain |
|---|---|---|---|
| deployed `iter_192` | 5555.73 (5561.58 at seed 0) | yes | +15.0 (+20.9) |
| best in-budget feasible (`iter_179`) | 5566.6 | yes | +25.9 |
| infeasible 5599.97 cluster | 5599.97 | **no** | +59.3 (do not claim) |

This is already acknowledged internally (`paper/REVIEWER_RESPONSE.md:175–184`
flags the ~5600 cluster as "all `train_feasible=False` … infeasibility"), and
the "5600 ceiling" wording survives only in **archived/superseded** files
(`paper/archive/sections/results.tex`, `paper/archive/sections/discussion.tex`,
and the working note `paper/APPLICATION_FACTS.md:116`); the live
`paper/sections/` contains no "5600" reference.

**Recommendation:** never headline "+59.3 on train". Use the feasible gain
(~+15 deployed, up to +25.9 best-feasible), and keep the AEP story secondary
to the feasibility story.
