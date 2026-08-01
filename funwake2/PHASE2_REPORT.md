# FunWake-2 — Phase 2 report (freeze + controller + DRY RUN)

**Scope: FREEZE + BUILD CONTROLLER + DRY RUN ONLY.** No Phase-3 launch, no
multi-generation real search, and **no real Claude/Gemini mutation spend** — the
dry run uses a deterministic MOCK mutator. STOP before Phase 3. All work is
additive under `funwake2/` (+ `funwake2/vendor/`). Nothing under `runs/`,
`archive/`, or `results/baselines*.json` was modified (verified by `git status`:
those paths are untouched; new files only).

Authoritative inputs: `specs/funwake2_spec.md` (incl. SIGN-OFF ADDENDUM),
`results/funwake2_prereg/PREREGISTRATION.md`, `funwake2/PHASE1_REPORT.md`
(R1–R4). Phase-1 accepted; skeleton_v2 carries the G8 float32-alpha0
canonicalization (native dei_n50 seed0 = 5560.7575185032, bit-identical across
fresh processes).

---

## 1. Re-measured post-G8 in-search c·D baseline (FROZEN)

Native seed (`funwake2/seeds/native.py`, lr scale 0.833·D → gamma_min),
`total_steps=8000`, `gamma_min=0.01`, **10 seeds (0–9)**, feasibility-gated, via
`funwake2/evaluator.evaluate`, into a FRESH dir `funwake2/state/g2_baseline/`
(the pre-existing s8000 checkpoints were PRE-G8 — e.g. old dei_n50 seed0 =
5560.314 vs the G8 canonical 5560.76 — and were NOT reused). Aggregated by
`funwake2/aggregate_g2.py` → `funwake2/controller/baselines_g2.json` (per-seed
AEP retained for paired scoring).

| cell | N | D | mean AEP (GWh) | std | sem | feasible | note |
|---|---|---|---|---|---|---|---|
| `dei_n50` | 50 | 240 | 5560.393 | 1.462 | 0.462 | 10/10 | stage-A smoke |
| `dei_n80_omnidir` | 80 | 240 | 8818.125 | 3.719 | 1.176 | 10/10 | |
| `dei_n120_rosedei` | 120 | 240 | 13029.988 | 7.496 | 2.370 | 10/10 | **HIGH-N** |
| `dei_n50_uniform` | 50 | 240 | 5598.413 | 1.315 | 0.416 | 10/10 | unidirectional |
| `parque_n20` | 20 | 80 | 231.505 | 0.688 | 0.218 | 10/10 | stage-A smoke |
| `parque_n30_uniform` | 30 | 80 | 356.171 | 4.884 | 1.544 | **0/10** | **INFEASIBLE reference** — see §7 flag; AEP used as a scale constant only |
| `parque_n10_omnidir` | 10 | 80 | 127.002 | 0.319 | 0.101 | 10/10 | |

(`funwake2/controller/baselines_g2.json`, `complete_10seed_all_cells=true`, per-seed
AEP retained for paired scoring. Post-G8: `dei_n50` seed0 = 5560.7575185032 was the
G8 canonical single-seed value; the 10-seed mean 5560.393 is the frozen reference.
`parque_n30_uniform` is genuinely infeasible under the c·D native — `max_sdf` 3–37 m
across seeds, 30 turbines do not pack into the Parque zones under a unidirectional
rose; this is the scale-constant-fitness case, flagged for sign-off in §7.)

These per-cell means are the **frozen in-search c·D baseline**; fitness is
`%-over-baseline` measured against them, paired by seed. Published in the
pre-registration (R4).

---

## 2. Frozen cells (D-4 finalized)

### 2a. Stage-B training set — 7 cells (FROZEN)
Registered in `funwake2/evaluator.CELLS` (`role="train"`, `stage_b=True`).
ROWP is in **NO** training cell (farm-level holdout).

| cell | farm | N | D | rose | note |
|---|---|---|---|---|---|
| `dei_n50` | DEI | 50 | 240 | DEI (own) | stage-A smoke cell |
| `dei_n80_omnidir` | DEI | 80 | 240 | omnidir | |
| `dei_n120_rosedei` | DEI | 120 | 240 | DEI | **HIGH-N (frozen)** — see §2d |
| `dei_n50_uniform` | DEI | 50 | 240 | uniform (dir=0) | unidirectional extreme |
| `parque_n20` | Parque (multizone) | 20 | 80 | DEI | stage-A smoke cell |
| `parque_n30_uniform` | Parque (multizone) | 30 | 80 | uniform | unidirectional |
| `parque_n10_omnidir` | Parque (multizone) | 10 | 80 | omnidir | |

Span: D ∈ {80, 240}, N ∈ {10…120}, roses {dei, uniform, omnidir}, incl.
multi-zone (Parque) + high-N (n120). Stage-A fast-reject cells = `dei_n50`,
`parque_n20` (2 cheap cells; **NOT** any high-N cell).

### 2b. Stage-B+ elite tier (gbar-only) — `dei_n200_rosedei`
`role="stage_b_plus"`, `gbar_only=True`. n200 is **NOT** in the per-generation
stage-B (≈357 s/eval; it tripped the watchdog / was impractical). It is the
elite-tier gbar cell: top-k archive elites × 2–3 paired seeds, run on gbar where
~5–6 min evals are affordable. `cascade.stage_b_plus()` **raises** unless
`enable_stage_b_plus=True`, so it can never execute inside the Mac evolution
loop.

**n200 classification — DEFERRED to gbar (NOT decided here).** A Mac 1-seed
probe (native@c·D, 8000 steps) was **infeasible** (boundary_penalty 1.18e-3 >
1e-3; min_dist 960.0) at 357 s — but that is ONE seed, not a classification. The
formal classification is a gbar native@c·D 5-seed run: if feasible → ordinary
stage-B+ elite cell; if infeasible → **capability-frontier** cell (moves into the
confirmatory test set). Left as a documented placeholder.

### 2c. Holdout + test set (frozen; AEP firewalled)
- **Holdout (selection only):** `rowp_n74` — ROWP farm-level holdout. AEP
  firewalled; used only for margin-aware multi-seed elite selection.
- **Frozen TEST set (touched once, at the deployment decision):**
  - `rowp_n200_roserowp`, `rowp_n300_roserowp` — **ROWP high-N**.
  - **Parque real heterogeneous wind resource** — the currently-unused
    per-cell WAsP Weibull A/k + speedup/turning maps. `problem_parqo.json` uses
    the homogeneous site-averaged surrogate; the heterogeneous problem JSON is a
    pre-test build step from `parqo/build_problem.py` (frozen by COMPOSITION
    here; **flagged** — see §7).
  - `rowp_n74_uniform` — **unidirectional extreme on an unseen farm** (ROWP
    polygon + uniform rose).
  - (If gbar classifies n200 infeasible, it joins the test set as the
    capability-frontier cell.)
  Test-cell AEP is never observed before the one-time 30-seed confirmatory test.
  High-N confirmatory cells are ONE-SHOT 30-seed runs where ~5-min evals are
  affordable — the evolution-loop ≤3-min limit does not apply to the test set.

### 2d. High-N stage-B decision — DEI n120 (timing curve)
The high-N stage-B cell was chosen from a native@c·D (8000-step, 1-seed) timing
+ feasibility sweep:

| N | seconds/eval | feasible | decision |
|---|---|---|---|
| 80 | 72 | yes | already a training cell |
| 100 | 116.8 | yes (min_dist 1098) | tractable but > ~90 s target |
| **120** | **159** | **yes (AEP 13016.6, min_dist 1037 > 960)** | **FROZEN as the high-N cell** |
| 150 | 228 | — | rejected (> ~3 min) |
| 200 | 357 | **no** (0/50 in the 500-MS baseline too) | → stage-B+ gbar-only |

n120 is the largest N with eval ≤ ~3 min AND a usable feasible native reference.

---

## 3. Per-cell texture floors (BLOCKING before freeze)

Method (G6): hold an optimized feasible native layout fixed, jitter turbine
positions at mm scale, measure AEP std (the wake-cone-mask discretization
texture) in absolute GWh.

| farm | per-cell floor (GWh) | basis |
|---|---|---|
| DEI | **0.3** | prior es_mechanism calibration (~0.005% of ~5500 GWh) |
| Parque | **0.1** | Phase-1 G6: AEP std 0.11 GWh @10 mm on parque_n20 |
| ROWP | **0.64** | measured here (`funwake2/gates/g6_rowp_floor.py`, rowp_n74): 10 mm-scale AEP std on the optimized native layout (base 4258.73 GWh); 1 mm → 0.233, 100 mm → 1.000 GWh |

The deployment criterion uses the **per-cell** floor for each test cell (not a
single 0.3). Post-G8, floors are NOT inflated by process noise (G8: same
(schedule,cell,seed) matches to 0.0000 across fresh processes), so the 30-seed
paired test keeps full power at these margins.

---

## 4. Evolutionary controller (chassis D-5 + our 3 custom parts)

### 4a. Vendored chassis (pinned fork)
`funwake2/vendor/openevolve/` — OpenEvolve pinned to commit
`411fb59c886c18704caaffb611e17cf9e7d824d2` (shallow clone, `.git` stripped to
avoid a nested repo; provenance in `funwake2/vendor/PIN.md`). Provides the
island MAP-Elites (`database.py`), cascade (`evaluator.py`), and prompt-sampler
scaffolding we build against. ShinkaEvolve's two eval-saving mechanisms are
**grafted** (not vendored wholesale) as ~100-line additions in
`funwake2/controller/novelty.py`: code-novelty rejection-sampling (embedding +
cheap-LLM dedup before a stage-B eval) and fitness/novelty-aware parent
sampling.

### 4b. File tree (`funwake2/controller/`)
```
config.py        RunConfig + FROZEN MAP-elites bins (D-8) + budget ceiling
cache.py         content-addressed (schedule_hash,cell,seed,gamma_min,steps) cache
lineage.py       append-only JSONL provenance (dedupe-on-resume; fsync/record)
cost.py          MAX_USD/MAX_TOKENS tracker + clean 90% abort
descriptors.py   MAP-elites descriptors from schedule_fn (pre-eval) + frozen binning
archive.py       per-island MAP-elites archive (serializable, atomic checkpoint)
novelty.py       ShinkaEvolve grafts (code-novelty rejection + parent sampling)
cascade.py       stage A / B / B+ (gbar) / C wrapping evaluator.evaluate
controller.py    evolutionary driver + checkpoint/resume + cost abort
run_dry.py       dry-run CLI (mock mutator; optional jax-free fake evaluator)
engines/base.py  Engine ABC + MutationLog + EvoContext
engines/mock.py  MOCK mutator (deterministic edits + synthetic tokens/cost)
engines/claude_sdk.py  Claude Agent-SDK adapter + ANTHROPIC_API_KEY preflight
engines/gemini_cli.py  Gemini CLI subprocess engine
seeds/seed_{native,cosine,cyclic}.py  self-contained scale-aware gen-0 seeds
tests/test_preflight.py  Agent-SDK preflight (raises on ANTHROPIC_API_KEY)
tests/test_machinery.py  archive/cost/cascade/fitness/firewall/resume tests
```

### 4c. Cascade (spec 3.2, `cascade.py`)
- **Stage A** — 2 cheap cells (`dei_n50`, `parque_n20`) × 2 seeds. Reject if any
  seed infeasible OR AEP < the per-cell c·D reference by more than the noise
  floor. NOT any high-N cell.
- **Stage B** — full frozen training matrix × ≥5 **paired** seeds (same seed ⇒
  same init as the reference, G8 bit-stable). `score_c = 100·(mean_seeds
  AEP_cand − mean_seeds AEP_ref)/AEP_ref` over the same seed subset. Aggregate =
  `mean_c(score_c)` with **worst-cell tiebreak** (`min_c score_c`). **Hard
  feasibility gate** = the CANDIDATE's own feasibility at gamma_min in every
  cell.
- **BLOCKING fitness patch:** `AEP_ref` is a **scale constant only** — used to
  make cells commensurable, valid even when the reference layout is itself
  infeasible (e.g. a capability-frontier cell). The candidate's feasibility is
  the independent hard gate; the reference's feasibility never gates. (Unit-
  tested: `test_fitness_scale_constant_with_infeasible_ref`.)
- **Stage B+** (elite-tier, **gbar-only**) — top-k archive elites × 2–3 paired
  seeds on the expensive high-N cells incl. n200; refuses to run unless
  `enable_stage_b_plus=True`.
- **Stage C** (elites only) — ROWP holdout margin (**AEP firewalled**: only
  feasibility booleans + a margin-over-floor boolean cross the firewall; raw
  holdout AEP is kept in a `_firewalled` block stripped before anything reaches
  a prompt) + the **gamma_min = 1.0 responsiveness** check (a schedule invariant
  to the tolerance is rejected).

### 4d. Mutation engines (`engines/`)
- **Claude Agent-SDK adapter** (`claude_sdk.py`) — **preflight RAISES if
  `ANTHROPIC_API_KEY` is present** (even empty — it shadows the OAuth profile),
  so a raw key can never shadow the subscription OAuth token and silently bill
  the metered API. Auth is the Agent-SDK path ONLY (`CLAUDE_CODE_OAUTH_TOKEN` →
  `claude_agent_sdk.query`); a raw-API `anthropic` client is never imported on
  this path (lazy import inside `mutate`). Logs model string + prompt/completion
  tokens + $. **Not invoked in the dry run.**
- **Gemini CLI engine** (`gemini_cli.py`) — `gemini -m … -p …` subprocess; logs
  model + tokens + $. **Not invoked in the dry run.**
- **MockEngine** (`mock.py`) — deterministic literal-perturbation + synthetic
  token/cost, deterministic in `(parent, gen, island, child_index)` so a resumed
  generation regenerates identical children. NO provider is ever contacted.

### 4e. MAP-elites archive (`archive.py`) — FROZEN bins (D-8)
`peak_lr/D {<0.5, 0.5–0.8, 0.8–1.2, >1.2}` · `terminal_lr_m {≤0.01, 0.01–0.1,
0.1–1, >1}` · `coupling {coupled, decoupled, cyclic}` · `restarts {0, 1–2, ≥3}`.
Per-island one-elite-per-cell; feasibility dominates, then mean fitness, then
worst-cell tiebreak. Gen-0 seeded with self-contained scale-aware ancestors
(native / iter192-family / iter118-family) that occupy **3 distinct cells** (R4;
unit-tested `test_archive_binning`: peaks 0.833 / 0.833 / 1.35, cyclic in the
>1.2 bin). Descriptors are computed from `schedule_fn` pre-eval.

### 4f. Lineage, checkpoint/resume, cost ceiling
- **Lineage** (`lineage.py`) — append-only JSONL, fsync/record: content hash,
  parent ID(s), engine + model string, prompt/completion tokens, $, walltime,
  descriptors, per-cell fitness (firewall-safe %-scores), generation, island,
  ancestor/port_transform, stage reached, status. Dedupe on `(candidate_id,
  generation)` so a resumed generation does not double-log.
- **Checkpoint/resume** (spec 5.3) — content-addressed result cache keyed by
  `(schedule_hash, cell, seed, gamma_min, steps)` (skip-existing on resume, zero
  recompute); archive + novelty state + cost + generation serialized atomically
  per generation; `run_id` recorded. Parent-sampling RNG is **derived** from
  `(run_id, generation)` (reconstructed, not stored) and parent lists are sorted
  by content hash → canonical; the mock engine and evals are deterministic, so a
  resumed generation reproduces the same candidates and hits the cache for every
  eval. Novelty seen-set is persisted (was the one non-derivable in-memory state)
  so novelty rejections reproduce.
- **Cost ceiling** (`cost.py`) — MAX_USD / MAX_TOKENS; cumulative tracked from
  the per-invocation engine logs; **clean abort at 90 % of either** (stop issuing
  mutations, finish in-flight, checkpoint, stop).

---

## 5. DRY RUN — machinery validation (MOCK mutator, NO LLM spend)

MOCK mutator (`MockEngine`), `--fake-eval` (jax-free deterministic stand-in) for
the state-machine checks; the real-eval cache/skip-on-resume path is covered by the
unit suite (`test_machinery.py`, 7/7). All scenarios via `funwake2/controller/run_dry.py`.

| scenario | command | result |
|---|---|---|
| **Full run** | `--fake-eval --run-all --gens 3 --proposals 3 --islands 2` | `STATUS=DONE gen=3`; archive 3 occupied cells; `BEST fitness=-0.0073 worst_cell=-0.0428` |
| **Cost accounting** | (same run) | `usd=0.0940/1000`, `tokens=20608`, `calls=9` — tracked from per-invocation mock logs |
| **Cascade / stage-A fast-reject** | (same run) | stage-A cells = `dei_n50` + `parque_n20` only (no high-N); non-negative fake Δ keeps feasible candidates un-rejected |
| **Archive binning (frozen bins, D-8)** | (same run) | gen-0 seeds occupy 3 distinct cells: `peak[0.8-1.2]` coupled/restart-0 & restart-1-2, and `peak[>1.2]\|term[>1]\|cyclic\|restart[>=3]` |
| **Lineage provenance** | (same run) | 12 append-only JSONL records: `candidate_id`, `parent_ids`, `engine`+`model`, prompt/completion tokens, `usd`, `descriptors`, `per_cell_fitness`, `generation`, `island`, `ancestor`/`port_transform`, `stage_reached`, `status` |
| **Resume bit-identity** | 3 × `--fake-eval` (1 gen/call, same `--state-dir`) vs single-process run-all | archive summary + BEST **identical**; lineage identical in **every** reproducible field — only the wall-clock `timestamp` differs (correct) |
| **Cost-ceiling abort** | `--fake-eval --run-all --gens 50 --max-usd 0.05` | `STATUS=ABORT aborted=True` at `usd=0.0525`; finishes the in-flight generation, checkpoints, stops |
| **Infeasible-reference fitness patch** | `--fake-eval --fake-infeasible-ref parque_n20` | `STATUS=DONE`, no crash, archive identical — `AEP_ref` used as a scale constant; reference feasibility never gates |

**Notes.**
- **No provider contacted, no spend, no network.** MockEngine is deterministic in
  `(parent, gen, island, child_index)`, so a resumed generation regenerates
  identical children; `claude_agent_sdk`/`anthropic` are never imported on the dry
  path.
- **Resume fidelity.** The single non-derivable in-memory state (the
  `NoveltyFilter` seen-set — the resume bug the build surfaced) is now persisted in
  the checkpoint; parent-sampling RNG is re-derived from `(run_id, generation)` and
  parent lists are content-hash-sorted, so resume needs no stored RNG. Confirmed:
  resumed vs run-all lineage differs **only** in `timestamp` (wall-clock write
  time); `candidate_id`, `content_hash`, `parent_ids`, `descriptors`,
  `per_cell_fitness`, tokens, `usd`, `stage_reached`, `status`, and even
  `walltime_s` are bit-identical, as are `archive.summary()` and `BEST`. The
  real-eval content-addressed cache (skip-existing on resume, zero recompute) is
  additionally exercised by `test_machinery::test_resume_bit_identity` and
  `test_resume_midgen_crash_bit_identity` (both pass).
- **Cost-abort semantics.** The 90 %-of-ceiling trigger stops *issuing* new
  mutations; the in-flight generation completes before checkpoint, so cumulative
  cost can cross 100 % by at most one generation's mock spend (here $0.0525 vs the
  $0.05 ceiling). By design ("stop issuing, finish in-flight, checkpoint, stop").
- **`--fake-eval` scope.** The fake evaluator (deterministic AEP from schedule
  bytecode hash) validates cascade order, archive binning, lineage, cost, and
  resume *instantly and chunk-free*; it deliberately bypasses the jax eval and its
  content-addressed cache (`CACHE hits=0 misses=0`), which the real-eval unit tests
  cover instead.

---

## 6. Agent-SDK preflight unit test

`funwake2/controller/tests/test_preflight.py` — **PASS**, no Claude invoked
(no `claude_agent_sdk`/`anthropic` import, no network). Verifies:
(1) `ANTHROPIC_API_KEY` set → `preflight()` raises `AnthropicApiKeyPresentError`;
(2) empty `ANTHROPIC_API_KEY=""` also raises (it shadows the OAuth profile);
(3) key unset + `CLAUDE_CODE_OAUTH_TOKEN` present → passes;
(4) key unset + OAuth token missing → raises `RuntimeError` (fails closed).

---

## 7. Flags — author decisions

- **`parque_n30_uniform` reference is INFEASIBLE (0/10) — NEW, decide at sign-off.**
  The post-G8 baseline sweep shows the c·D native cannot make `parque_n30_uniform`
  feasible: `max_sdf` 3–37 m across all 10 seeds (turbines meters outside the
  Parque zones), not a mm-scale artifact — 30 turbines simply do not pack into the
  Parque zones under a unidirectional rose (contrast `parque_n20`/`parque_n10` at
  ~0.01 m SDF). The design is **robust to this** (the scale-constant fitness patch,
  §4c: `AEP_ref` normalizes cells but never confers feasibility credit; the
  candidate's own feasibility at gamma_min is the independent hard gate), but an
  infeasible reference was anticipated only for the n200 capability-frontier cell,
  **not** for a stage-B training cell. **Decision:** (a) **retain** it as a hard
  stage-B cell — a candidate reaching feasibility there does something the baseline
  cannot, which is informative; or (b) **swap** it for a feasible lower-N Parque
  unidirectional cell (e.g. `parque_n20_uniform`) so the unidirectional-Parque slot
  has a feasible reference. The frozen cell set is NOT changed pending this call;
  the baseline table records the reference faithfully as infeasible either way.
- **n200 classification (gbar):** run native@c·D 5 seeds on gbar to classify
  n200 as a stage-B+ elite cell (if feasible) or a capability-frontier test cell
  (if infeasible). The Mac 1-seed probe was infeasible but is not authoritative.
- **Parque heterogeneous test cell:** the real heterogeneous WAsP resource is
  frozen by COMPOSITION but its problem JSON must be BUILT (from
  `parqo/build_problem.py`) as a pre-test step; not yet materialized.
- **Real-engine smoke (pre-launch):** a 1–2 live-mutation smoke on Claude
  (after `claude setup-token`) and Gemini (after Gemini auth) is the remaining
  pre-launch step; NOT done here (no real spend).
- **gbar env standup:** stage-B+ / high-N confirmatory runs need the clean-room
  env on gbar (Phase-1/3 setup task).

---

## 8. Confirmation

- **NO Phase-3 launch.** No multi-generation real search; no real Claude/Gemini
  mutation was invoked (dry run = MockEngine only).
- Nothing under `runs/`, `archive/`, or `results/baselines*.json` was modified.
- Pre-registration updated + frozen: `results/funwake2_prereg/PREREGISTRATION.md`
  (per-cell G2 baselines + per-cell floors filled).
