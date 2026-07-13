# Claim audit — every comparative claim, marked against the artifacts

Verified 2026-07-13 by opening the files. Legend:
**V** = verified-from-artifact · **W** = weakened/contradicted-by-a-file-we-opened · **N** = not-run / no-artifact.

## Provenance (A)

| claim | mark | evidence |
|---|:--:|---|
| Claude deployed = iter_192 | **V** | best-val feasible-both (val 4271.5); `results_agent_schedule_only_5hr/iter_192.py` exists |
| Gemini deployed = iter_118 | **V** | best-val feasible-both (val 4269.3); `results_agent_gemini_cli_5hr/iter_118.py` exists |
| A5 "Claude iter_118" | **W** | copy-paste error — should be 192 |
| Claude betas constant (0.3, 0.5) | **V** | printed from script: β1 min=max=0.300, β2 min=max=0.500 |
| Gemini betas cyclic | **V** | printed: β1∈[0.10,0.39], β2∈[0.20,0.878] |
| low-beta finding is about Claude's iter_192 | **V** | `ab_stdbetas.py` derives from iter_192; `gemini_iter192_fixed_betas.py` is an unrelated older file (different purpose), NOT used |

## Random-search control (B)

| claim | mark | evidence |
|---|:--:|---|
| family includes cosine + gaussian-bumps (incl. dual) | **V** | `random_search_ablation.py` L72 (cosine), L85+L99 (gaussian_bumps, n_bumps∈{1,2}) |
| family specified AFTER the LLM runs | **W** | committed 2026-04-13 (`b08a4fea`); Claude run Apr-5, Gemini Apr-7. **Generous/circular control — disclose.** |
| sampler self-contained (no CLAUDE.md/Ideas/seed reads) | **V** | only reads own attempt-log; `import os` for paths |
| champion selected by feasibility-filtered best-val-AEP | **V** | I filtered train_feasible∧rowp_feasible then max rowp_aep (LLM protocol); so N=300 infeasibility is a real generalization failure, not circular |
| 320 samples, count-matched to Claude's 320 | **V** | `--n-samples 320`; 320 in log. Count-matched, NOT compute-matched |
| family CAN reach low β2 (0.5) but champion didn't (0.77) | **V** | family β2∈[0.5,0.9999]; champion params β2=0.7677 |

## Feasibility table (C) — ROWP rose, feasible restarts %, [strict-0 / 0.1m / 1m / 5m]

Source: `ms_highn_v2/rowp_n{N}_roserowp.json` (baseline/claude/gemini),
`random_scale/*.random.json` (random), `es_baseline/*_es_seed*.out` (ES).

| schedule | N=200 | N=300 |
|---|---|---|
| naive baseline | 0 / 0 / 48 / 100 | 0 / 0 / 0 / 12 |
| ES baseline | 94 / 96 / 100 / 100 | **72 / 82 / 84 / 88** |
| Claude iter_192 | 6 / 100 / 100 / 100 | **0 / 54 / 96 / 100** |
| Gemini iter_118 | 96 / 96 / 98 / 100 | **44 / 56 / 70 / 100** |
| random champion | 4 / 14 / 82 / 100 | **0 / 0 / 10 / 100** |

**Reconciliation (V):** Claude at N=300 is **0% strict-0** and **54% at 0.1m** — both numbers real, different columns.

## Headline comparative claims

| claim | mark | evidence |
|---|:--:|---|
| random matches LLM on AEP (N=300: +2.14 vs +2.10) | **V** | `ms_highn_v2`, @5m matched |
| LLM beats random on feasibility at 0.1m at scale (54% vs 0%) | **V** | C table, 0.1m column |
| "LLM schedules are strict-feasible at scale" | **W** | **CONTRADICTED — Claude 0% strict-0 at N=300.** True only at 0.1m |
| "LLM matches the production ES baseline" | **W** | **CONTRADICTED — ES BEATS the LLM at every tolerance** (72/82/84/88 vs Claude 0/54/96/100) |
| low betas (0.3,0.5) are the feasibility mechanism; std → 0% | **V** | `fig_ablation`; ab_stdbetas at N=200/300 |
| "Claude *discovered* low betas" | **W** | **hinted** — Ideas list L213 "Standard Adam (0.9,0.999) vs TopFarm (0.1,0.2)". Mechanism is ours (ablation); choice was prompted |

## Missing / untested (N)

| item | mark | note |
|---|:--:|---|
| early stopping applied ON TOP of the LLM schedules (N=300 strict-0) | **N** | **NOT RUN.** Matrix skeleton has no ES. ES is schedule-agnostic (`sgd.py:122`, triggers on lr_i/lr_0 ≤ threshold) → it would bolt onto iter_192. **Likely the decisive experiment.** |
| AEP gains at finer sector resolution | **N** | computed at **12 sectors**; not retested at 36. Coarse sectors inflate layout gains |
| replicated reruns (3 seeds × 2 agents) | **N** | `run_reruns.sh` configured, **never launched**; 0/6 done. All model & single-schedule claims are **n=1** |

## Bottom line

Three headline claims are **contradicted or weakened** by the files: the LLM is *not*
strict-feasible at scale (Claude 0%), it does *not* beat the production ES baseline
(ES dominates), and the low-beta choice was *hinted*. Two decisive checks are
**unrun**: ES-on-LLM-schedules, and any n>1 replication. What survives verified:
random matches AEP; the LLM beats random + naive baseline on 0.1m-feasibility at
scale; the low-beta *mechanism* is real. The honest paper is "the LLM finds
schedules competitive with production practice and better than naive/random
controls," pending the ES-on-LLM experiment and the reruns.
