# Counterfactual: rotor-diameter-scaled lr0 on the validation farm

**Question.** The deployed schedule search (`runs/schedule_only_5hr`, 316
`iter_NNN.py` files, deployed: `iter_192.py`) evaluated ROWP with the
skeleton's hardcoded `lr0 = 50.0`. On the D=240 m training farm that is
50/240 = 0.208 rotor diameters. If ROWP (D=198 m, confirmed from
`results/problem_rowp.json`) had used the same fraction of *its* rotor
diameter -- `lr0 = 50 x 198/240 = 41.25 m` -- would the deployment choice
change? **Train evaluation is unchanged by design** (DEI stays at lr0=50;
the attempt-log train scores stand as-is).

## Answer: yes -- the deployment choice changes

The historical deployment rule (confirmed from the log and README) is
**best ROWP AEP among train-feasible AND ROWP-feasible** candidates. At
lr0=50 it selects `iter_192.py` (ROWP 4271.49 GWh) -- reproduced exactly.

| | lr0=50 (actual) | lr0=41.25 (counterfactual) |
|---|---|---|
| Winner | `iter_192.py` -- 4271.49 | **`iter_181.py` -- 4272.49** |
| Runner-up | `iter_228.py` -- 4271.48 | `iter_334.py` -- 4271.51 |
| Margin over runner-up | 0.01 GWh (below noise) | 0.98 GWh (~2x the 0.5 GWh noise floor) |
| `iter_192` under this arm | rank 1 | 4265.20, rank 91 of 244 feasible |
| `iter_181` under this arm | 4263.43, mid-pack | rank 1 |

- Realized validation performance of the deployment: 4271.49 -> 4272.49,
  i.e. RD-scaling the validation lr0 would have improved the deployed
  outcome by **+1.00 GWh** (twice the noise floor, but modest).
- Keeping `iter_192` while switching to lr0=41.25 would have *hurt*:
  -6.29 GWh (4265.20). The old winner does not survive the change.
- Schedule family: `iter_181` is the **same lineage** as `iter_192`
  (5x lr-coupled alpha + quadratic late ramp + narrow alpha-dip
  "explore-then-enforce", beta1=0.3, beta2=0.5) but with a **pure cosine
  LR and no LR bumps**, versus `iter_192`'s dual Gaussian LR bumps. The
  family survives; the specific LR-bump variant does not.
- Under the *stated* alternative rule (best train AEP among
  ROWP-feasible): winner is `iter_179.py` at both lr0=50 and 41.25 --
  unchanged -- but that rule does not reproduce the historical `iter_192`
  deployment, so the ROWP-AEP rule above is the operative one.

## Aggregate shifts (ROWP, 50 -> 41.25; all 266 legit train-feasible files)

- **Feasibility rate:** 231/248 = 93.1% at lr0=50 (known flags) vs
  244/266 = 91.7% at 41.25. Paired flips: 4 lost (`iter_064/066/134/278`),
  7 gained (`iter_010/087/130/203/279/303/318` -- incl. `iter_303`,
  train 5562.21, a top-15 train file), 237 stable; flip rate 11/248 = 4.4%.
  Of 18 files with no lr0=50 flag, 10 are feasible at 41.25.
- **AEP shift (248 paired):** mean +1.10 GWh (4259.53 -> 4260.63), median
  +1.53. Among the 227 feasible-in-both: mean +1.07 +/- 5.30 GWh, range
  -14.19 to +17.42; 131 files move up >0.5 GWh, 82 move down >0.5 GWh.
  Small mean benefit, large per-schedule scatter.
- **Rank correlation (Spearman, ROWP AEP 50 vs 41.25):** rho = 0.556
  (p ~ 2e-21). The ranking is substantially reshuffled -- which is why the
  winner changes even though the mean shift is ~1 GWh.
- **Old top-5 train cluster** (`iter_179/166/177/183/165`): all remain
  ROWP-feasible at 41.25. ROWP AEPs move -5.3 to +3.7 GWh; best is
  `iter_166` (4270.12, rank 14). None reaches the top of the 41.25 ranking.

## Method notes

- **Tool:** `tools/reeval_lr0.py` replicates the
  `run_optimizer.py --schedule-only` -> `harness.py` -> `skeleton.py`
  pipeline exactly, with `lr0` parameterized (skeleton.py:102) and the
  `alpha0 = mean|grad_obj|/lr0` coupling (skeleton.py:103) preserved.
  Feasibility identical to `benchmarks/dei_layout.ProblemBenchmark`
  (boundary_penalty < 1e-3 AND min dist >= 0.99*min_spacing).
  `tools/reeval_lr0_driver.py` ran the sweeps (6-way concurrent,
  checkpointed, 0 failed evals; the deciding `iter_181` value was
  repeat-run: 4272.49 both times).
- **Validation:** at lr0=50 the tool reproduces log values exactly for
  train (`iter_099` 5557.38, `iter_123` 5565.31, `iter_165` 5566.24,
  `iter_166` 5566.57) and ROWP (`iter_192` 4271.49, `iter_179` 4263.34,
  `iter_166` 4266.43, `iter_181` 4263.43; `iter_216` infeasible, matching
  its log flag). Four files (`iter_177/179/192/228`) miss their
  *original-run* train values by 4-6 GWh but exactly reproduce the log's
  own later current-stack re-scores (e.g. `iter_179` -> 5560.90 = late
  attempt 323; `iter_192` -> 5561.58 = late attempt 335): the historical
  train numbers carry cross-stack drift (pre-vendored pixwake) for a
  minority of bump-sensitive schedules. Train ranking is nevertheless
  taken from the log as-is, per the experiment definition. Note the
  log's `rowp_aep` fields were backfilled post-hoc on the *current*
  stack, so the ROWP comparison (the part that decides deployment) is
  stack-consistent on both arms.
- **Attempt-to-file mapping:** `iter_NNN.py` == attempt NNN for NNN <= 236
  (identity -- also the convention `tools/cleanse_and_backfill.py` itself
  used). Built by pairing the 316 original-window train entries
  (timestamp before the 11-day gap, 5000 < train_aep < 5700) in
  timestamp order with the 316 files in number order; file numbering has
  gaps at 237-255 and 288-290. Verified by the exact reproductions above.
  Saved: `mapping_iter_to_attempt.json`.
- **Log anomaly (excluded entries):** `attempt_log.json` holds 423
  attempts vs 316 files. Attempts 321-423 (timestamps ~11 days after the
  run) are post-run generalization/N-sweep matrix evals -- their
  train_aep (2318-8830 GWh) spans the `problem_dei_n30..n100` farms
  (values above the N=50 farm's ~6570 GWh capacity), while
  train_baseline stays 5540.72 because `run_optimizer.py` always
  attaches the farm-1 baseline. Four original-window entries (attempts
  307/309/311/313, AEP 4240-4600) are ROWP-scored entries interleaved in
  the train log. All excluded from the deployment analysis.
- **Cancelled arm:** `train/` contains 126 partial lr0=240 DEI evals from
  the pre-scope-change experiment (kept as incidental data only; the
  corrected question leaves train at lr0=50).

## Files

- `rowp/*.json` -- all 266 ROWP evals at lr0=41.25
- `validation/*.json` -- lr0=50 reproductions + `iter_181` repeat
- `train_lr50/*.json` -- 9 current-stack DEI lr0=50 reference evals
- `summary.json`, `mapping_iter_to_attempt.json`,
  `candidates_trainfeasible.txt`
