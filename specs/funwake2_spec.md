# FunWake-2 — Design Spec (PHASE 0, for author sign-off)

Evolutionary discovery of the best **deployable** TopFarm-SGD schedule under a
**scale-aware** parameterization. This is an ENGINEERING search: seeding the
population with known-good schedules is encouraged. Publication-grade provenance
is mandatory so any later "the system discovered X" claim is auditable against
lineage.

**Status: DESIGN ONLY — no search compute.** Phases 1 (build + smoke-test on the
incumbent), 2 (finalize pre-registration), 3 (launch) each require explicit
author go-ahead. This document + `results/funwake2_prereg/PREREGISTRATION.md` are
the Phase-0 deliverables.

**Relationship to funwake_v2/:** the single-farm fair-baseline clean-room
(built separately) is the seed of this; FunWake-2 generalizes it to a scale-aware
multi-farm evolutionary search and reuses its firewall discipline and c·D-tuned
baseline. The v1 lr0 investigation (tuning/matrix/paired/diameter-rule, EDITS,
audit, iter_ archives) stays firewalled from every mutator workspace.

**Ground rules (enforced everywhere):** all additive; nothing under `runs/`,
`archive/`, or `results/baselines*.json` is modified. The mutator's workspace
contains the harness, seeds, and per-generation feedback ONLY — never the audit
narratives or paired-test reports. Holdout/test AEP values never enter any
mutator context or transcript (key-only checks, as in v1).

---

## 1. Scale-aware schedule interface (proposal to refine)

### 1.1 Signature
```python
def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    """Return (lr, alpha, beta1, beta2) at `step`.
       D          : rotor diameter (m)           — the exploration scale source
       min_spacing: min inter-turbine spacing (m) — packing scale
       n_turbines : N                             — density/packing signal
       gamma_min  : constraint tolerance (m)      — TopFarm's absolute terminal-lr,
                                                    the ONLY user-supplied scale
       alpha0     : penalty normalizer, skeleton-computed (§1.3)
    """
```
- **No free `lr0` anywhere.** Exploration lr must be constructed from `D` inside
  the schedule (e.g. `lr_peak = c * D`, `c` a schedule-internal constant/curve —
  evolvable, but never a user knob). This is the structural fix for the v1
  confound (a per-farm lr0 that had to be tuned).
- **`gamma_min` is the terminal learning rate in metres = the constraint
  tolerance** (TopFarm's own convention: the final lr is a proxy for how tightly
  the boundary is satisfied). It is the single user-supplied scale; the schedule
  decays lr toward it.
- Old `lr0` and `total_steps` semantics preserved otherwise; `total_steps` stays
  a fixed harness constant per run (proposal: 6000 = TopFarm native horizon; open
  decision D-1).

### 1.2 Skeleton changes (fixed skeleton; agent controls only schedule_fn)
The frozen skeleton must (a) pass `D, min_spacing, n_turbines, gamma_min` from
the problem JSON into `schedule_fn`; (b) compute `alpha0` per §1.3; (c) run the
Adam loop unchanged. No hardcoded lr anywhere. This is an additive skeleton
variant (`playground/skeleton_v2.py`), leaving v1 skeleton untouched.

### 1.3 `alpha0` normalization (proposal)
`alpha0 = mean(|∇_x J|, |∇_y J|) / D` at the initial (wind-aware grid) layout,
computed by the skeleton. Rationale: v1 used `mean|∇J|/lr0`, which coupled the
penalty scale to the (now-removed) lr0; normalizing by `D` keeps the penalty
scale problem-intrinsic and scale-covariant. The schedule composes `alpha(t)`
from `alpha0` (TopFarm-equivalent coupling is `alpha(t)=alpha0·D/lr(t)`, i.e. the
native `mean|∇J|/lr(t)`; decoupled/cyclic variants are the evolvable space). Open
decision D-2: confirm `/D` vs `/min_spacing` vs `/(D·√N)` — the smoke test §1.5
will show which keeps the native port on its known score.

### 1.4 Incumbent ports (VALIDATED BEFORE ANY SEARCH — the fidelity gate)
All ports logged as **seeded ancestors** with explicit lineage (`ancestor=native|iter_192|iter_181|iter_118`, `port_transform=…`).
- **(a) Native TopFarm** at `lr_scale = 0.833·D` decaying to `gamma_min`,
  ported to the new signature. **GATE:** it must reproduce the
  `results/lr0_diameter_rule/` baseline@0.833·D scores within the measured noise
  floor on every cell present there (DEI≈5561.3, ROWP≈4261.7, Parque≈231.1,
  10-seed means; ≤0.3 GWh / within SEM). If it does not reproduce, the port
  (or the §1.3 normalization) is wrong and the search does NOT proceed. This is
  the single go/no-go gate for Phase 1.
- **(b) `iter_192`, `iter_181`, `iter_118`** rescaled by `(0.833·D)/50` (their lr
  profiles were designed against lr0=50; this maps them onto the D-scale). Not
  gated on performance — logged as diverse seeded ancestors to populate the
  MAP-elites archive across behavioral bins from generation 0.

### 1.5 Phase-1 smoke test (before Phase 2/3)
Run the native port (a) on the fidelity-gate cells and confirm reproduction;
run ports (b) once per training cell for archive seeding; confirm `alpha0`
normalization choice (D-2). Deliver a one-page fidelity report. No search yet.

---

## 2. Fitness and farm split (proposal, subject to the stated constraints)

### 2.1 Fitness
Per training cell *c*, over the SAME paired seed set for candidate and baseline:
```
score_c = 100 * ( mean_seeds AEP_candidate,c  −  mean_seeds AEP_cDbaseline,c )
                 / mean_seeds AEP_cDbaseline,c          # % over the c·D-tuned native baseline
feasible_c = (all paired seeds satisfy the gamma_min gate)   # HARD GATE
```
- `cDbaseline` = the native port (a) at 0.833·D on cell *c*, same seeds
  (paired). Expressing fitness as %-over-baseline-per-cell makes cells of
  wildly different AEP magnitude (DEI ~5500 GWh vs Parque ~230) commensurable.
- **Feasibility at `gamma_min` is a hard gate:** any cell with an infeasible
  paired seed ⇒ candidate fails that cell (score_c = −∞ / excluded from archive
  as infeasible). No AEP credit for infeasible layouts.
- **Cross-cell aggregator (proposal):** `fitness = mean_c(score_c)` with a
  **worst-cell tiebreak** (`min_c score_c` breaks ties within the texture floor).
  Rationale: reward broad scale-generalization, but do not let a big win on one
  easy cell mask a regression on a hard cell. Open decision D-3: mean% vs
  worst-cell-primary vs CVaR (mean of worst-k).

### 2.2 Farm split (must span ≥2 rotor diameters; hard cells included)
- **Training cells (span D and difficulty):**
  - DEI (D=240): `n50·rosedei`, `n80·roseuniform` (unidirectional extreme +
    high-N packing).
  - Parque (D=80): `n20` and a high-N packing cell, **multi-zone** (disconnected
    inclusion zones — the hardest feasibility geometry).
  - One mid-scale/omnidirectional cell for coverage (e.g. DEI `n50·roseomnidir`).
  - → proposal: 5–6 training cells spanning D∈{80,240}, N∈{20…80}, roses
    {dei, uniform, omnidir}, incl. multi-zone + high-N. (Full grid available:
    `results/matrix/problem_{dei,rowp}_n{30..300}_rose{dei,omnidir,rowp,uniform}.json`
    + `parqo/problem_parqo.json`.) Open decision D-4: final cell list.
- **Farm-level holdout (margin-aware multi-seed selection, never single-eval
  argmax):** ROWP (D=198) — an intermediate scale NOT in training. Used ONLY to
  select among elites by validation-mean margin (§4 selection rule). Its AEP is
  firewalled from mutators.
- **Pre-registered TEST set (touched exactly once, at deployment decision):**
  candidates — **ROWP high-N** (n200/n300), **Parque's real heterogeneous wind
  resource** (currently unused anywhere — the strongest OOD test), and the
  **unidirectional extremes** on an unseen farm. Final test-set frozen in the
  pre-registration (§4). Its AEP is never observed until the one-time test.
- **Firewall:** holdout + test AEP are computed by a separate scorer process
  whose numeric AEP outputs are key-checked out of any string that reaches a
  mutator prompt/transcript (feasibility-boolean + cell-id may pass; AEP may
  not). Same key-only discipline as v1's ROWP firewall.

---

## 3. Controller: fork an open evolutionary chassis (do NOT rebuild the v1 loop)

### 3.1 OpenEvolve vs ShinkaEvolve — evaluation
| need | OpenEvolve | ShinkaEvolve |
|---|---|---|
| islands + migration | ✅ island MAP-Elites + periodic migration | ✅ island-based |
| MAP-elites on CUSTOM descriptors | ✅ configurable feature dims from evaluator metrics (direct fit for §3.4) | partial — novelty is embedding-based; custom bins less first-class |
| evaluation cascade | ✅ built-in cascade w/ timeouts/retries/artifacts | supports staged eval; less turnkey |
| multi-parent / evidence prompting | ✅ prompt sampler curates parent + lineage + artifacts | ✅ fitness/novelty-aware parent sampling |
| novelty / duplicate rejection | implicit (MAP-elites bins) | ✅✅ **headline**: embedding+LLM code-novelty rejection-sampling |
| **sample efficiency** (evals cost 30–50 s) | good w/ our cascade | ✅✅ designed for it — "thousands→hundreds" evals |
| lineage w/ parent IDs | ✅ evolution history + ancestors in prompts | ✅ tree with parent IDs |
| LLM backend | OpenAI-compatible ensemble / model-based islands | LLM ensemble (UCB1 adaptive) |
| maturity for our exact feature list | higher (AlphaEvolve-complete, configurable metrics) | newer, research-grade |

**Recommendation: fork OpenEvolve as the chassis, graft ShinkaEvolve's two
eval-saving mechanisms.** OpenEvolve meets the hard requirements natively —
**custom MAP-elites descriptors** (§3.4), **built-in cascade** (§3.2), per-island
model assignment (Claude-island vs Gemini-island, §3.3), and lineage-in-prompt.
The one thing it lacks — the aggressive sample-efficiency that matters because
each eval is 30–50 s — is addressed by (i) our own stage-A fast-reject cascade
and (ii) porting ShinkaEvolve's **code-novelty rejection-sampling** (embedding +
cheap-LLM dedup BEFORE spending a stage-B eval) and **fitness/novelty-aware
parent sampling**, both portable ~100-line additions. Neither framework natively
speaks "Claude Agent-SDK-over-OAuth" or "Gemini CLI subprocess," so a custom LLM
adapter is required either way (§3.3) — that cost is a wash. **Open decision D-5:
if the Phase-1 smoke test shows eval budget is the binding constraint even with
cascade+novelty-rejection, switch the chassis to ShinkaEvolve.** Fork, don't
depend: vendor the chosen repo under `funwake2/vendor/` pinned to a commit.

We supply three custom parts:

### 3.2 Evaluator — cascade wrapping `tools/run_optimizer.py`
- **Stage A (fast-reject):** 2 cheap cells (e.g. DEI n50·rosedei, Parque n20) ×
  2 seeds. Reject if infeasible or score < seeded-native by more than the noise
  floor. ~4 evals (~3 min).
- **Stage B (full):** the full training matrix (§2.2) × ≥5 **paired** seeds
  (same init layouts as the per-cell c·D baseline). Compute §2.1 fitness. ~30–40
  evals (~20–25 min).
- **Stage C (elites only):** holdout (ROWP) margin check — multi-seed validation
  mean vs the c·D baseline, feasibility-gated, AEP firewalled. Only archive elites
  reach C.
- Each stage wraps `tools/run_optimizer.py --schedule-only` (v2 harness).
  Feasibility at `gamma_min` enforced at every stage. Artifacts (feasibility,
  behavioral descriptors, per-cell %) captured for prompt context; **AEP of
  holdout/test never captured into prompt context.**

### 3.3 Mutation engines (log model string + token/cost per invocation)
- **(i) Claude via the Agent SDK, billed to the Max-plan Agent SDK credit.**
  Auth via `claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN`. **Hard requirement:
  verify `ANTHROPIC_API_KEY` is UNSET in the engine's environment** so it cannot
  shadow the OAuth token and silently bill raw API. **Never wire a raw-API
  Anthropic client to the subscription OAuth token** — use the Agent SDK path
  only. Preflight assertion in the adapter: refuse to start if `ANTHROPIC_API_KEY`
  is present. Log the resolved model string per call.
- **(ii) Gemini via the existing Gemini CLI** (the v1 `gemini -p` subprocess
  path). Log model string + tokens.
- Per-island model assignment (OpenEvolve model-based islands): Claude-islands
  and Gemini-islands, so lineage records which engine produced each mutation.
- Every mutation logs: parent ID(s), engine, model string, prompt-token/
  completion-token counts, $ cost estimate, wall-time, and the diff.

### 3.4 MAP-elites archive — behavioral descriptors + bins
Descriptors computed from the candidate's `schedule_fn` (cheap, pre-eval):
| descriptor | definition | bins (proposal) |
|---|---|---|
| `peak_lr_over_D` | max_t lr(t) / D | [0,0.4),[0.4,0.7),[0.7,1.0),[1.0,1.5),[1.5,2.5),[2.5,∞) |
| `terminal_lr_m` | lr(total_steps−1), metres | [<0.005),[0.005,0.02),[0.02,0.1),[0.1,0.5),[≥0.5) |
| `alpha_coupling_class` | coupled(α∝1/lr) / decoupled / cyclic | 3 categorical |
| `restart_bump_count` | # local lr maxima (peaks) over the run | 0,1,2,3–5,6–9,≥10 |
Archive = MAP-elites over the 4-D grid (per island). Descriptors are logged with
every candidate for the provenance/lineage record and to seed diverse ancestors
(§1.4b span these bins). Bin edges frozen in the pre-registration.

---

## 4. Pre-registration (draft delivered: `results/funwake2_prereg/PREREGISTRATION.md`)
- **Selection (holdout):** deployed = elite with max ROWP validation-mean margin
  over the c·D baseline (≥5 seeds), feasibility-gated; **never single-eval
  argmax**; margin must exceed the 0.3 GWh texture floor.
- **Deployment criterion (test set, one-time):** the deployed schedule beats the
  c·D-tuned native baseline on the pre-registered TEST set under the **paired
  30-seed protocol** — 95 % t-CI excludes 0 AND |mean Δ| > 0.3 GWh — with
  **strict feasibility at `gamma_min`**. Both outcome paragraphs written verbatim
  in the pre-registration (SUCCESS / NULL).
- **Deployment-fidelity stage (mandatory before any "deployable" claim):** the
  winner is re-validated in **real TopFarm / py_wake with stochastic wind
  sampling** via the `ex_dei.py` harness (ported to each test farm), not just the
  pixwake surrogate — to confirm the schedule survives the true stochastic
  optimizer it will deploy into. A schedule that wins on pixwake but degrades
  under real-TopFarm stochastic sampling is NOT deployable.

---

## 5. Budget, parallelism, resume, cost ceiling

### 5.1 Eval budget (per candidate, ~40 s/run)
| stage | runs | wall (serial) |
|---|---|---|
| A fast-reject | 2 cells×2 seeds = 4 | ~3 min |
| B full | ~6 cells×5 seeds = 30 | ~20 min |
| C holdout (elites) | ROWP×5 = 5 | ~3.5 min |
Most candidates die at A. Assume per generation: 20 proposals → ~6 pass A → B →
~2 archived elites → C. Gen cost ≈ 20·(A) + 6·(B) + 2·(C) ≈ 80 + 120 + 7 ≈
~207 min serial ≈ 3.5 h/gen. Target ~30 gens ⇒ ~105 h **serial** eval time.

### 5.2 Parallelism plan
- **Local:** the eval is embarrassingly parallel over (cell × seed). Fan out
  ~8–12 concurrent runs → ~8–12× ⇒ ~9–13 h wall for 30 gens locally (disk-aware;
  prior sweeps filled disk once — monitor).
- **gbar (recommended for the full run):** LSF array jobs, one array task per
  (candidate × cell × seed). Requires the funwake_v2 clean-room env stood up on
  gbar first (pixi/pixwake — a Phase-1 setup task; gbar reachable, LSF `bsub`,
  no repo/env there yet). With ~100-way arrays the eval wall collapses to the
  slowest single candidate chain (~25 min/gen). **Open decision D-6:** local for
  Phase-1 smoke + small runs; gbar for the Phase-3 full run.

### 5.3 Checkpoint / resume (every long run here has died ≥once — resume must be trivial)
- Program database + MAP-elites archive + island state + RNG serialized to
  `funwake2/state/` after every generation (atomic write + prev-kept).
- Every eval result is a content-addressed JSON keyed by
  `(schedule_hash, cell, seed)`; re-launch skips any key already present → a
  killed run resumes at the exact missing evals, zero recompute.
- Lineage/token/cost logs are append-only JSONL, fsync'd per record.
- `funwake2 resume <state-dir>` reloads and continues; verified in Phase-1.

### 5.4 Agent-SDK credit sizing + cost ceiling
- Mutations ≈ 20/gen × 30 gens = ~600, split Claude/Gemini (say 60/40 → ~360
  Claude Agent-SDK calls). Est. ~8–20k tokens/mutation (parent code + evidence +
  completion) ⇒ ~3–7 M Claude tokens over the run. Size against the Max-plan
  monthly Agent-SDK credit BEFORE launch; if a single run exceeds a fraction
  (proposal: ≤50 %) of monthly credit, split across billing periods or shift more
  mutation share to Gemini CLI.
- **Hard cost ceiling + abort rule:** set `MAX_USD` and `MAX_TOKENS` for the run;
  the controller tracks cumulative $ and tokens (from the per-invocation logs)
  and **aborts cleanly at 90 % of either ceiling** (finishes in-flight evals,
  checkpoints, stops issuing mutations). No run may silently exceed budget.
  Proposal defaults: `MAX_USD` = author-set; `MAX_TOKENS` from §5.4 estimate ×1.5.

---

## Open decisions for author sign-off
- **D-1** total_steps (6000 native vs 8000).
- **D-2** alpha0 normalizer (`/D` vs `/min_spacing` vs `/(D·√N)`) — smoke-test-settled.
- **D-3** cross-cell aggregator (mean% + worst-cell tiebreak vs worst-cell-primary vs CVaR).
- **D-4** final training cell list (§2.2).
- **D-5** chassis: OpenEvolve+graft (recommended) vs ShinkaEvolve — revisit after Phase-1 eval-efficiency.
- **D-6** compute: local vs gbar for the full run (+ gbar env standup as Phase-1 task).
- **D-7** final TEST set composition (ROWP high-N / Parque heterogeneous wind / unidirectional extreme).
- **D-8** MAP-elites bin edges (§3.4) frozen into the pre-registration.

## Sources (chassis evaluation)
- ShinkaEvolve — Sakana AI, arXiv 2509.19349, github.com/SakanaAI/ShinkaEvolve (Apache-2.0).
- OpenEvolve — pypi.org/project/openevolve, algorithmicsuperintelligence.ai/blog/openevolve-overview.

---

# SIGN-OFF ADDENDUM (Phase-0 approved with corrections) — 2026-07-31

## Decisions (final)
- **D-1** total_steps = **8000 uniform**. Gate the 6000-step native extraction vs
  `lr0_diameter_rule` (proves extraction faithful); then ONE **bridge run**
  promotes an 8000-step native variant to the **in-search c·D baseline**.
- **D-2** `alpha0 = mean|∇J|/D`, AND the native TopFarm port computes alpha0
  from **D** — NOT the driver default `mean|∇J|/lr` (they differ by **1.2×** at
  c=0.833: `mean|∇J|/(0.833D) = 1.2·mean|∇J|/D`). The `/lr` form MUST NOT ship.
- **D-3** cross-cell = **mean% + worst-cell tiebreak**, per-cell feasibility a
  hard gate.
- **D-4** **6–8 cells** spanning D=80–240, incl. **≥1 high-N** and **≥1
  multi-zone**; heterogeneous-wind Parque **reserved for test** (not training).
- **D-5** approved (fork OpenEvolve, graft ShinkaEvolve novelty-rejection).
- **D-6** **local** for smoke; **gbar** for the full run.
- **D-7** test set as proposed, PLUS a **stage-C elite check at gamma_min = 1.0 m**
  — schedules must demonstrably RESPOND to the tolerance input (a schedule whose
  behavior is invariant to gamma_min is rejected).
- **D-8** MAP-elites bins FROZEN: `peak_lr/D` {<0.5, 0.5–0.8, 0.8–1.2, >1.2};
  `terminal_lr_m` {≤0.01, 0.01–0.1, 0.1–1, >1}; `coupling` {coupled, decoupled,
  cyclic}; `restarts` {0, 1–2, ≥3}.

## BLOCKING CORRECTION — incumbent port scale (was 4× too hot)
The v0 spec's incumbent rescale `(0.833·D)/50` is **4× too hot**: iter_192's peak
is ~4× its internal lr0, so `(0.833·D)/50` puts its peak at ~3.3·D on DEI — the
regime the paired data shows breaking feasibility.
- **Fix:** incumbent internal scale = **0.2083·D (= 50·D/240)**, which keeps
  iter_192's peak at ~**0.83·D** (4 × 0.2083·D). (The native baseline separately
  uses lr_scale 0.833·D because it *starts* at its scale; incumbents *ramp up* ~4×,
  so their scale is 4× smaller.)
- **Phase-1 gate (bit-identity):** ported iter_192 at D=240 must be
  **bit-identical** to archived `iter_192@lr0=50` (0.2083·240 = 50 **exactly**),
  given the same alpha0 — validates the port transform (lr0→0.2083·D, algebra
  unchanged).
- **Documented:** this anchors the ports to DEI (D=240 → 50). On ROWP (D=198) the
  incumbent operating point shifts to lr0 = **41.25** (0.2083·198); the 41.25
  counterfactual delta was already measured in the v1 ROWP re-eval.

## funwake_v2/
Build-only, **search launch deprecated.** Its clean-room + native-seed extraction
+ fair baseline (DEI 5561.6 / ROWP 4262.1) fold into FunWake-2 as Phase-1 assets.

---

# n200 DECISION + FITNESS PATCH (Phase-2) — 2026-07-31

**n200 removed from per-generation stage-B** (measured >5 min/eval; D-4 placement
underestimated cost). n80 does NOT inherit the high-N role — hard-instance
pressure is the point.

**Stage-B high-N cell (chosen by measurement, point 3): DEI n120/dei rose** —
159 s/eval (≤3 min), feasible reference (native AEP 13016.6, min_dist 1037).
Timing: n80 72s, n100 116s, n120 159s ✓, n150 228s ✗ (all feasible). Files:
`funwake2/problems/problem_dei_n{100,120,150}_rosedei.json`.

**New cascade tier — stage-B+ (elite-tier):** top-k archive candidates/generation
only, 2–3 paired seeds, **gbar only** (never in the Mac evolution loop). n200 is a
stage-B+ cell; its cost is added to the budget table.

**BLOCKING FITNESS PATCH (§2.1 amended, applies to ALL cells):** the per-cell
normalization does NOT assume the reference baseline is feasible.
`score_c = 100·(AEP_cand − AEP_ref,c)/AEP_ref,c` uses `AEP_ref,c` purely as a
**scale constant** — valid even when the reference layout is infeasible at that
cell. The **candidate's own feasibility at gamma_min remains the independent hard
gate** (infeasible candidate ⇒ cell fails, regardless of score). So a cell whose
reference is infeasible (possibly n200) is still a valid fitness cell: it
normalizes the candidate's AEP without conferring feasibility credit.

**n200 classification (point 5, gbar-only — never on Mac):** native@c·D, 5 seeds,
gamma_min=0.01. If FEASIBLE → ordinary stage-B+ elite cell. If INFEASIBLE → the
**capability-frontier cell**: a candidate achieving strict feasibility there is a
qualitatively new result, and it enters the confirmatory test set. Deferred to
the gbar env standup (Phase-1/3 setup task).

**Ops (point 6):** long-eval cells get per-seed heartbeat logging + a raised
(configurable) watchdog; any n200-class run is gbar-only.

**Test set (point 7):** unchanged; high-N confirmatory cells are ONE-SHOT 30-seed
runs where ~5-min evals are affordable — the ≤3-min evolution-loop cost limit does
NOT apply to the confirmatory test set.

---

# PHASE-2 REVIEW ADDENDUM (accepted-contingent items 1–6)

**Item 1 — `parque_n30_uniform` 0/10 reconciliation (BLOCKING, resolved: NO v2 bug).**
The c·D native is 0/10 feasible on `parque_n30_uniform` (`max_sdf` 3–37 m). This is
NOT a multizone fidelity gap: skeleton_v2's `_init_positions`, `multizone_penalty`,
and `multizone_sdf` are the SAME function objects as the vetted `skeleton_multizone`
(proven by `is` identity + seed-0 init max|Δ|=0), and the old schedule at lr0=50 run
through skeleton_v2 reproduces the genuine old-pipeline regime (2/10 feasible incl.
seed 5; per-seed point values diverge only through the documented G8 float32-vs-float64
alpha0 chaos). The reference baselines the review cited are incomparable: the
`parqo_native_ms` "12/12 feasible" is a **best-of-K=50 multistart** (only 2/50
single-run feasible on `uniform|n30`), and the diameter-rule "9–10/10" was **Parque
N=20, DEI rose**. The infeasibility is a genuine unidirectional-rose × tight-zone ×
scale effect (recorded as a finding). Evidence: `funwake2/state/diag_n30/`.

**Item 2 — SWAP.** `parque_n30_uniform` → **`parque_n14_uniform`** in stage-B. A probe
of N∈{28…10} (c·D native, T=8000) showed N≥16 is a chaotic feasibility knife-edge
(N16 3/4, N24 0/4) while N=14/12/10 are 4/4; N=14 is the LARGEST uniform-Parque N with
a **10/10** c·D reference (10 seeds, mean 184.81 GWh). It preserves Parque×unidirectional
coverage with an all-feasible reference. `parque_n30_uniform` → capability-frontier tier
(`role="capability_frontier"`, gbar-only), beside `dei_n200_rosedei`.

**Item 3 — stage-B global hard gate (FROZEN).** Stage-B keeps the **global per-cell
hard gate**: a candidate must be feasible in EVERY stage-B cell (`cascade.stage_b`:
`feasible_all = AND_c sc.feasible`; one infeasible cell ⇒ the whole candidate is
rejected). This is coherent ONLY with all-feasible references — the item-2 swap
guarantees all 7 stage-B references are 10/10 feasible. **Capability-frontier cells
(`dei_n200_rosedei`, `parque_n30_uniform`) are elite-tier informational and NEVER
gate** — they are scored (scale-constant fitness) but excluded from the hard gate.
Locked in by unit test `test_candidate_infeasible_one_stage_b_cell_fails` (a candidate
feasible everywhere but one stage-B cell FAILS stage B).

**Item 4 — workspace-scoping LAUNCH GATE (built; smoke auth-blocked).**
`funwake2/controller/workspace.py`: each mutation runs with cwd in a freshly
materialized clean-room dir containing ONLY `INTERFACE.md` + sanitized harness/seeds/
parent + firewalled `feedback.json`; `results/`, `paper/`, `specs/`, the prereg,
audit docs, `funwake2/state/`, `baselines_g2.json`, and `evaluator.py` are OUTSIDE
the readable tree. Enforced by cwd (Gemini `subprocess(cwd=)`, Claude
`ClaudeAgentOptions(cwd=)`) + `allowed_tools=[]`. `sanitize()` strips seed docstrings +
redacts residual forbidden tokens; `assert_clean()` RAISES on any leak; `scan_tree()`
re-greps the post-run transcript. Unit-tested (`test_workspace_scoping.py`, 4 tests).
The live 2-mutation smoke is ENV-BLOCKED on BOTH engines: Claude (no
`CLAUDE_CODE_OAUTH_TOKEN`, `claude_agent_sdk` not installed → `pip install
claude-agent-sdk` + interactive `claude setup-token`) and Gemini (installed CLI
returns `IneligibleTierError`, unsupported individual-tier client → migrate/update).
The gate code is complete + unit-tested; the smoke + transcript grep runs pre-launch
once an engine is usable.

**Item 5 — test-cell floors.** Re-measured at test-set freeze on the actual test cells
(`rowp_n200`/`rowp_n300` do NOT inherit n74's 0.64 GWh). Deferred to that step.

**Item 6 — Parque heterogeneous test JSON: BUILT** locally —
`parqo/problem_parqo_hetero.json` (`build_problem_hetero.py`), preserving the
(12×20×20) per-cell WAsP maps. No gbar needed. gbar remains the long pole for n200
classification + the full Phase-3 run. **Launch holds until items 1–4 land and the
prereg reflects the final stage-B set** (done for 1–3 + item-4 build; item-4 smoke
pending auth).

---

# ROUND-2 REVIEW ADDENDUM (saturation + launch-prep)

**R2-1 — `parque_n14_uniform` is FEASIBILITY-ONLY (saturation confirmed).** 14 ×
single-turbine free-stream AEP under the uniform rose = 184.8117 GWh = the optimized
n14 baseline (deficit −0.0003 GWh ≪ the 0.1 GWh Parque floor;
`funwake2/state/diag_n30/n14_saturation.json`). The objective is SATURATED: under
dir=0 + the hard 2σ wake cone, every all-escape feasible layout scores exactly
free-stream (std=0, no texture, no AEP signal). n14 is therefore a **feasibility-only
stage-B gate** — `CELLS[...]["feasibility_only"]=True`; the hard feasibility gate is
retained but its ~0 % score is EXCLUDED from the mean-%/worst-cell aggregate
(`cascade.stage_b`; 6 of 7 cells scored). Unit test
`test_feasibility_only_cell_excluded_from_aggregate`.

**R2-2 — n30 reconciliation postscript.** The reviewer-cited "12/12 feasible" is
CELL-level best-of-K=50 (12 rose×N cells each had ≥1 feasible start), not per-run;
per-run feasibility on `uniform|n30` was always ~4 % (2/50). No fidelity gap was
implied. `funwake2/state/diag_n30/` diagnostics are retained as reusable fidelity
assets.

**R2-3 — Claude-first launch; Gemini restoration is parallel/non-blocking.** Phase-3
launches on the Claude Agent-SDK engine alone. Gemini restoration (updated CLI/auth,
or a metered-API engine on Google credits) is non-blocking. Archival note: v1's
Gemini CLI tier (individual Gemini Code Assist) is deprecated upstream
(`IneligibleTierError`), which is why the v1 `gemini -p` path no longer authenticates.

**R2-4 — Heterogeneous-Parque PRE-TEST evaluator gate (prereg-frozen).** Before the
heterogeneous Parque test cell is unblinded, the heterogeneous evaluator must be built
AND validated: fed the site-averaged climate it must reproduce the homogeneous
surrogate's AEP within the Parque floor (0.1 GWh) or a documented expected offset. A
heterogeneous evaluator that fails this degenerate check is a bug; its test result
would be uninterpretable.

**R2-5 — `assert_clean` ast-parses sanitized reference code.** The scoping gate now
`ast.parse`s every .py in the materialized scope, so a `sanitize()` REDACTED
substitution can never hand the mutator syntactically broken reference code (fail-
closed). Tests: `test_sanitized_reference_code_parses`, `test_assert_clean_raises_on_broken_py`.

**Phase-3 launch review = saturation resolved (R2-1 ✓) + Claude smoke clean (pending
auth) + prereg current (✓).** gbar standup (n200 classification + full run) remains
the long pole.

---

# ROUND-3 REVIEW ADDENDUM (methodology-draft fixes, pre-pilot)

**R3-1 — FARM-BALANCED aggregate (resolves D-3).** The cross-cell aggregate is
`fitness = mean over farms of (mean over that farm's scored cells of score_c)`, so
each farm contributes equally irrespective of its cell count (the training set has 4
scored DEI cells vs 2 scored Parque cells). Worst-cell tiebreak (`min_c score_c`) and
the hard feasibility gate are unchanged. Implemented in `cascade.stage_b`
(`_cell_farm`); parity unit test `test_farm_balanced_aggregate_parity` (a uniform +1%
across a farm's cells raises fitness by 1%/n_farms, equal across farms). Companion
items: **feasibility-only cells run at 2 seeds** (they gate feasibility only); and
**gbar-only capability-frontier cells are PENDING off gbar** (deferred to the elite
tier, never gating) — `test_gbar_only_cell_pending_off_gbar`.

**R3-2 — Stage A is a GROSS fast-reject.** Reject a (cell,seed) iff INFEASIBLE at
gamma_min OR AEP more than `stage_a_reject_frac` (~1%) below the reference — NOT
texture-floor-tight (a floor-tight Stage A would mass-reject the QD exploration the
archive exists for). Texture floors are used for selection margins only.
`StageResult.causes` tallies rejection reason (ok / infeasible / below_ref / error) as
the pilot Stage-A-rejection-rate-by-cause metric. Test
`test_stage_a_gross_filter_and_causes` (a 0.5%-below candidate PASSES).

**R3-3 — Reproducibility scope.** Same-platform: evaluation is bit-identical across
independent processes (float32-canonicalized alpha0, G8). Cross-platform: agreement to
within the measured drift tolerance (v1 arm64/x86: ±1.7 GWh). The framework is a
deterministic function of seeds/config/budget on a fixed platform; cross-platform
results agree within that drift.

**R3-4 — Training resources / test set / floors / bib.** Training spans THREE resource
types (DEI rose, omnidirectional, unidirectional); the ROWP rose is held-out only. The
frozen TEST set is ROWP high-N (`rowp_n200_roserowp`, `rowp_n300_roserowp`), the Parque
real heterogeneous resource (`problem_parqo_hetero.json`), and the ROWP unidirectional
extreme (`rowp_n74_uniform`). Per-cell texture floors: DEI 0.3, Parque 0.1, ROWP 0.64
GWh. Bibliography: cite Quick et al. **2022** (WES Discussions) consistently with the
companion paper.

**R3-5 — Capability-frontier tier (named).** `dei_n200_rosedei` and
`parque_n30_uniform` are the capability-frontier tier: reference-infeasible cells
tracked at the elite tier as qualitative probes (a candidate reaching strict
feasibility there is a qualitatively new result), never gating the per-generation
search.
