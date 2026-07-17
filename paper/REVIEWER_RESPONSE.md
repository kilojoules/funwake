# Reviewer response + experiment action plan — WES_Funwake-12

## TL;DR (revised after running the controls)

- **Most requested experiments already exist / are now run**: matched-budget
  random search over the schedule family (**scaled to N=300**), the production
  early-stopping baseline, alpha/beta/bump ablations. The only genuinely-new
  agent experiment is **replicated reruns** (Major #2, infra ready).
- **The corrected #1 result FAVORS the LLM.** Random search over the schedule
  family ties the LLMs on *raw AEP at loose tolerance* — but its **strict-tolerance
  feasibility collapses with scale** (~40% at N=80 → **0–14%** at N=200/300),
  while the LLM schedules hold **54–100%**. So the LLM's advantage is not raw
  AEP; it is **discovering schedules that stay feasible at deployable tolerance
  as N grows** — the constraint work the no-early-stopping skeleton no longer
  does. See `fig_random_control` (Panel A concedes AEP parity; Panel B shows
  the feasibility collapse). An earlier read that "random wins" was an artifact
  of comparing AEP at 5 m tolerance, which erases exactly this advantage.
- **#3 is confirmed and softens a different claim.** The production
  (early-stopping) baseline is 72–92% feasible at strict-0 (vs ~0% no-ES), so
  Fig 5's feasibility gap was largely a no-ES-baseline artifact; the LLM
  **matches** the production baseline (having discovered the constraint-satisfying
  structure without being given early stopping).

### Superseded pre-control note (kept for the record)

- The matched-budget random search over the schedule family
  **essentially ties the LLMs on final AEP** (val 4268.6 vs Claude 4271.5,
  Gemini 4269.3; baseline 4243.6) *at N=74 and at loose tolerance*. Taken alone
  this looked like the reviewer's worst case; the tolerance-resolved,
  scaled result above overturns it. The paper's central claim reframes from
  *"LLMs beat search on AEP"* to *"LLMs
  discover the search space / structure; cheap search matches once the space
  is known."*

---

## Major points

### #1 Non-LLM search baseline at matched budget — DONE, favors the LLM

**FINAL VERDICT (tolerance-resolved, scaled to N=300):** random search matches
the LLM on AEP but its **strict-tolerance feasibility collapses at scale**
(0.1 m: ~40% at N=80 → 0–14% at N=200/300; LLM holds 54–100%). The LLM wins on
**self-feasibility at scale**, not raw AEP. Dedicated figure: `fig_random_control`
(Panel A = AEP parity at 5 m; Panel B = feasibility collapse at 0.1 m). Random
line removed from `fig_aep_dominance` to avoid implying parity without the
feasibility caveat. Caveat to disclose: the random family *includes* the
LLM-discovered motifs (gaussian-bumps, cosine) — a generous control that still
fails on feasibility, which only strengthens the point.

---

**Status: experiment done** (`tools/random_search_ablation.py`,
`results_random_search_320/`, described in `discussion.tex:167`).

- **Family** (sampled a priori): cosine / exponential / linear / polynomial LR
  decay, optional warmup (0/2/5/10 %), perturbation ∈ {none, sinusoidal,
  Gaussian bumps}, monotonic penalty ramps (linear/quadratic/coupled), random
  Adam betas. **320 samples = matched to Claude's attempt count.**
- **Result** (best feasible-on-both, ROWP val AEP):

  | method | val AEP | train AEP | reached at |
  |---|--:|--:|--:|
  | random search (N=320) | **4268.6** | 5561.8 | attempt 31 |
  | Claude (deployed 192) | 4271.5 | 5555.7 | attempt 192 |
  | Gemini (deployed 118) | 4269.3 | 5565.1 | attempt 118 |
  | 500-start baseline | 4243.6 | 5545.0 | — |

- **Implication**: the LLM edge over random search is **+0.07 % (2.9 GWh),
  within multistart noise**, and random search reached its best *earlier*
  (attempt 31 vs 118/192). We **cannot** claim LLMs beat dumb search on final
  AEP.

**Honest reframe (recommended)** — the defensible claims are:
1. **Automated discovery works**: both LLM and random search beat the
   hand-designed baseline (+0.6 % val, and +constraint-precision).
2. **The LLM's contribution is search-space definition, not point search.**
   The random-search family already *contains* the LLM-discovered motifs
   (`decay_type="cosine"` = Gemini restarts; `perturbation="gaussian_bumps"` =
   Claude dual-bump). We could only design that family *after* seeing the
   discoveries. The LLM found the structure de novo, from raw code, with no
   parameterization handed to it.
3. **LLM output is interpretable and transferable** (dual-bump / cyclic-beta
   motifs), which random parameter vectors are not.

**Actions:**
- (a) Promote random search from a discussion paragraph to a **results
  figure/line** (overlay random-search best-so-far on Fig 2, or a small
  companion panel). This preempts the reviewer instead of burying it.
- (b) Rewrite abstract / intro / conclusions to the reframe above. Delete any
  "LLMs outperform search" phrasing.
- (c) **Optional strengthening experiment** — random search over a *naive*
  family (a priori, *excluding* gaussian-bump / cosine) to show the family
  itself is the LLM's contribution. If the naive family underperforms, that is
  the cleanest evidence of LLM value. ~1 h compute (320 evals, reuse harness).
- (d) Optional: Optuna/CMA-ES over the **schedule** parameters (our existing
  CMA-ES is layout-level, `n_dim=148` — wrong axis, don't cite it here).

### #2 Replicated runs — NOT done; the real new experiment

**Status: n=1 per agent. Genuinely missing.**

- Rerun **≥3 seeds per agent** with: explicit `--model` pin
  (`claude-opus-4-5-*`, `gemini-3-flash-preview`), `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`
  (memory off — closes the audit gap), `--output-format json` / retained
  session transcripts, and logged CLI versions.
- Cost: 5 h × 3 seeds × 2 agents ≈ **30 agent-hours** + API credits. Cheap for
  the credibility.
- Payoff: converts Fig 2 from an anecdote to a **distribution**; lets us either
  *make* the Claude-vs-Gemini comparison or *cleanly disclaim* it (currently
  n=1 vs n=1 — should not be read as a model comparison, and we should say so).
- **Priority: HIGH.**

### #3 Early-stopping (production) baseline — DONE, confirms the reviewer

**FINAL VERDICT:** ran the production ES baseline (K=50, rowp × {uniform,rowp,
omnidir} × N=50/80/200/300). It is **72–92% feasible at strict-0** on
rowp/omnidir (vs ~0% no-ES) — at N=300 it beats both LLM schedules on
feasibility. So Fig 5's feasibility gap was **largely a no-ES-baseline
artifact**. Added as a green dashed line in `fig_feasibility`. Honest reframe:
the LLM **matches** the production baseline (both strict-feasible + similar AEP)
having discovered the constraint-satisfying structure without early stopping;
the constraint-precision *advantage* is over the naive baseline and random
search, not the production one. Exception: unidirectional N=300 (ES ~8%), where
the schedules genuinely win.

---

**Status: ES implementation built + validated** (`validation/early_stopping/`:
pixwake ES vs TopFarm2, bit-for-bit tests, threshold 0.1). **Not yet a third
baseline line in Figs 4–5.**

- The reviewer is right that the Fig 4–5 baseline is the *weakened* no-ES seed
  schedule (constraint violations ~ final LR). Practitioners run TOPFARM with
  early stopping, which largely fixes feasibility.
- **Action**: run the baseline matrix **with early stopping** (threshold 0.1,
  K=50) and add it as a **third line** in Figs 4–5. This directly tests whether
  the feasibility gap in Fig 5 is an artifact of benchmarking the weakened
  variant. If ES closes most of the feasibility gap, we report that honestly
  and pivot Fig 5's message to *AEP-at-matched-feasibility*.
- Cost: baseline-only matrix rerun with ES ≈ same as one matrix baseline pass
  (few hours gbar). **Priority: HIGH.**

### #4 Ablate the discovered schedules — consolidated; scale-ablation recommended

**CONSOLIDATED FINDING:** the alpha-ablation (`results_alpha_ablation/`) sweeps
the penalty weight 0.01×–10× and val AEP moves **< 0.3%** (4260–4272 GWh), all
feasible **at N=74**. So the schedule components (alpha coupling/terminal slam,
Gaussian bumps, betas) are **not about AEP** — they do **constraint enforcement**,
which is invisible at the deployed scale but decisive at N=300 (see #1: the
LLM's feasibility holds at scale, random's collapses). **Recommended follow-up
(machinery ready in `validation/stochastic_aep/schedules_ablation.py`:
`no_bumps`, `alpha_scaled`):** run iter_192 vs {no-bumps, no-terminal-alpha-slam,
constant-betas} through the K=50 matrix **at N=300** and measure strict
feasibility — this isolates *which* component gives Claude its feasibility-at-scale.
That is the scientifically valuable version of Fig 3's ablation, and it directly
explains the #1 result. ~1 slow gbar cell per variant.

---

**Status: partial.** Have alpha-ablation (`results_alpha_ablation/`),
fixed-betas (`results/ablations/gemini_iter192_fixed_betas.py`), bump ablations
(`lumi/ablation_bump_de.sbatch`, `validation/stochastic_aep/schedules_ablation.py`).

- Fig 3 currently has a **placeholder caption** — this is the scientific
  payload.
- **Action**: consolidate into a transplant/ablation table: dual-bump vs
  single-bump (Claude), constant-low betas vs standard, the t=0 alpha spike,
  Gemini's cyclic betas vs constant. Write the Fig 3 caption. May explain the
  ParqueFicticio failures (test the alpha-spike / bump structure on the
  V80 5-zone geometry).
- **Priority: MEDIUM** (machinery exists; needs consolidation + write-up).
  Lower than #1–#3 if compute is tight, as the reviewer notes.

### Aside — the ~5600 GWh cluster near attempt 270 (Claude) — EXPLAINED

- Attempts ≈262–268 with train AEP ≈5600 are all **`train_feasible=False`**
  (constraint-violating layouts that score high AEP by pushing turbines out of
  bounds / under spacing). Validation-based selection requires feasibility on
  both farms → correctly passed them over. **Not overfitting or a scorer
  exploit — infeasibility.**
- **Action**: one sentence in the Fig 2 caption; confirm they render as
  **hollow** (infeasible) markers, not filled. (The value clusters near 5600
  but is not an exact sentinel.)

---

## Inconsistencies to reconcile (text fixes)

| # | Issue | Fix |
|---|---|---|
| a | Sec 2.4 says test = "best of 500 initial guesses"; Fig 4 caption says "max of 50 starts" | Two different procedures: the final generalization **test** uses 500 TopFarm starts; the per-cell **matrix** (Fig 4) uses the matched **K=50** multistart. State both distinctly (now in `methods.tex`, "Baselines" ¶). |
| b | max-of-N is a noisy, right-tail-flattering statistic | Report **distribution / mean-of-top-k with uncertainty** (add spread to Figs 4–5, or a supplementary), not bare max. |
| c | Fig 3 legend "Claude iter 192" vs Appendix A5 "Claude iter_118" | A5 typo (copy-paste of Gemini's 118). Fix to **iter_192**. |
| d | A5: deployed families "not named in either hint set" — but Ideas list has "Cosine annealing of lr with warm restarts" (= Gemini's schedule) | **Soften**: cosine restarts *are* in the hints; only Claude's **dual Gaussian bump** is absent. Reword to claim novelty for the dual-bump only. |
| e | Sec 2.4 title "Verification" vs "validation" in text | Standardize to **validation** throughout. |
| f | Test site: "Parque de Fictitio" / "ParqueFicticio" / "Parque Fictitio" / "Parqo" (Fig 1 axis) | Standardize to **"Parque Ficticio"** (or ParqueFicticio); fix the Fig 1 density-axis "Parqo". |
| g | Search-loop feasibility tolerance not stated | The agent's `run_optimizer` judged feasibility at `boundary_penalty < 1e-3` (≈ cm) AND spacing ≥ 0.99·d_min — the **same soft criterion as the baseline**. State it in Sec 2.4. |

---

## Do we need to run more experiments? — Action plan

| item | new run? | compute | priority |
|---|---|---|---|
| #2 Replicated reruns (3 seeds × 2 agents, pinned/memory-off/logged) | **YES** | 30 agent-h + API | **HIGH** |
| #3 Early-stopping baseline as 3rd line (Figs 4–5) | **YES** | few h gbar | **HIGH** |
| #1c Naive-family random search (fair LLM-value test) | optional | ~1 h | MED-HIGH |
| #4 Ablation consolidation + Fig 3 | mostly write-up | small | MED |
| #1a/b Surface random search + reframe claims | no | — | **HIGH (writing)** |
| #5 5600-cluster sentence + inconsistencies a–g | no | — | HIGH (writing) |

**Recommended sequence:** (1) reframe the central claim around the
already-in-hand random-search result and surface it in Fig 2 — highest
credibility-per-effort, no compute; (2) launch reruns (#2) and the ES baseline
(#3) in parallel on gbar; (3) consolidate ablations + write Fig 3 caption while
those run; (4) sweep the text inconsistencies. The only items that *require*
new compute are #2, #3, and optionally #1c.
