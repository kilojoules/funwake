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
| `parque_n14_uniform` | 14 | 80 | 184.812 | 0.000 | 0.000 | 10/10 | **SWAP** (see §1a) — Parque×unidirectional |
| `parque_n10_omnidir` | 10 | 80 | 127.002 | 0.319 | 0.101 | 10/10 | |

(`funwake2/controller/baselines_g2.json`, `complete_10seed_all_cells=true`, per-seed
AEP retained for paired scoring. Post-G8: `dei_n50` seed0 = 5560.7575185032 was the
G8 canonical single-seed value; the 10-seed mean 5560.393 is the frozen reference.
**All 7 stage-B references are 10/10 feasible** — the coherent-hard-gate precondition
(item 3). `parque_n14_uniform` is a **FEASIBILITY-ONLY** gate (round-2 item 1, §1b):
its objective is saturated, so it is in the hard gate but **excluded from the mean-%
aggregate** — 6 cells are scored.)

### 1a. SWAP: `parque_n30_uniform` → `parque_n14_uniform` (item-1 reconciliation)

The originally-frozen `parque_n30_uniform` had a **0/10-feasible c·D reference**
(`max_sdf` 3–37 m). Investigation (`funwake2/state/diag_n30/`) established this is
**NOT a v2 multizone fidelity bug**, so the swap (not a launch block) is warranted:

- The genuine old TopFarm multistart pipeline (`parqo_native_ms`, native_schedule,
  T=6000) was itself only **2/50 single-run feasible** on `uniform|n30`; its "12/12
  feasible" is the **best-of-K=50 multistart**, not per-run. The diameter-rule
  "9–10/10" reference was **Parque N=20, DEI rose** — a different, easier cell.
- skeleton_v2's multizone init/penalty/SDF are the **same function objects** as the
  vetted `skeleton_multizone` (proven: `is` identity; seed-0 init max|Δ|=0). Running
  the old schedule at **lr0=50** through skeleton_v2 reproduces the old 2/10 regime
  (2/10 feasible incl. seed 5). Per-seed point values diverge only through the
  documented alpha0 chaos (skeleton_v2 float32-canonicalizes alpha0 per G8; the old
  pipeline used raw float64).
- The c·D reference infeasibility on N=30-uniform is therefore a **genuine
  unidirectional-rose × tight-zone × scale effect**, recorded as a finding.

**Reconciliation postscript (item 2).** The reviewer-cited "12/12 strictly feasible"
is **cell-level** feasibility — 12 (rose × N) cells each had ≥1 feasible start out of
K=50 (best-of-50). The **per-run** feasibility rate on `uniform|n30` was always
~**2/50 ≈ 4 %**; skeleton_v2's 0/10 single-run is the same regime. No fidelity gap was
ever implied. The `funwake2/state/diag_n30/` diagnostics (same-object proof,
lr0=50-through-v2 reproduction, N-sweep, saturation check) are retained as reusable
**fidelity assets** for future multizone changes.

### 1b. `parque_n14_uniform` is a FEASIBILITY-ONLY gate (round-2 item 1)

n14's baseline std is exactly 0 because the objective is **saturated**:
`14 × single-turbine free-stream AEP = 184.8117 GWh = the optimized baseline`
(deficit **−0.0003 GWh**, ≪ the 0.1 GWh Parque floor;
`funwake2/state/diag_n30/n14_saturation.json`). Under dir=0 + the hard 2σ wake
cone, every all-escape feasible layout scores exactly free-stream — no mask
crossings, no texture, no AEP-improvement signal. So n14 is reclassified
**feasibility-only**: `CELLS["parque_n14_uniform"]["feasibility_only"]=True`; the
**hard feasibility gate is retained** (a candidate must reach a feasible Parque ×
unidirectional layout) but its ~0 % score is **excluded from the mean-%/worst-cell
aggregate** (`cascade.stage_b`; 6 cells scored). Locked by
`test_feasibility_only_cell_excluded_from_aggregate` (aggregate excludes n14; the
hard gate still fails a candidate infeasible there).

Per the swap rule ("largest uniform-Parque N with an all-feasible c·D reference"):
a probe of N∈{28,26,24,22,20,18,16,14,12,10} (T=8000) showed N≥16 is a chaotic
knife-edge (N=16→3/4, N=24→0/4) while **N=14/12/10 are 4/4 with interior margins**.
N=14 is the largest, confirmed **10/10** at 10 seeds (mean 184.81 GWh). It fills the
Parque×unidirectional stage-B slot with an all-feasible reference. `parque_n30_uniform`
moves to the **capability-frontier tier** (§2b), informational, never gating.

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
| `parque_n14_uniform` | Parque (multizone) | 14 | 80 | uniform | **FEASIBILITY-ONLY** (swap ← n30, §1a; saturated, §1b) |
| `parque_n10_omnidir` | Parque (multizone) | 10 | 80 | omnidir | |

Span: D ∈ {80, 240}, N ∈ {10…120}, roses {dei, uniform, omnidir}, incl.
multi-zone (Parque) + high-N (n120). All 7 references 10/10 feasible (coherent
hard gate, item 3). **6 cells enter the mean-% aggregate**; `parque_n14_uniform`
is a feasibility-only gate (§1b). Stage-A fast-reject cells = `dei_n50`,
`parque_n20` (2 cheap cells; **NOT** any high-N cell).

### 2b. Capability-frontier / elite tier (informational, NEVER gating)
Two cells sit outside the stage-B hard gate (item 3: frontier cells are
elite-tier informational, never gating):

- **`parque_n30_uniform`** (`role="capability_frontier"`, `gbar_only=True`) —
  CONFIRMED capability-frontier: c·D native is 0/10 feasible (§1a). A candidate
  that reaches **strict** feasibility here is a qualitatively new result. Scored
  informationally (AEP_ref = scale constant); never enters the stage-B feasibility
  gate.
- **`dei_n200_rosedei`** (`role="stage_b_plus"`, `gbar_only=True`) — see below;
  classification deferred to gbar.

#### Stage-B+ elite tier (gbar-only) — `dei_n200_rosedei`
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

### 4g. Workspace-scoping LAUNCH GATE (item 4) — firewall containment
`funwake2/controller/workspace.py`. Before a mutation, `materialize()` builds a
fresh clean-room dir (outside the repo tree) containing ONLY: `INTERFACE.md`
(schedule signature), a sanitized `skeleton_v2.py`, sanitized `seeds/`, the
sanitized `parent.py`, and firewalled `feedback.json` (%-scores + feasibility
booleans; raw AEP rejected). Everything else — `results/`, `paper/`, `specs/`,
the pre-registration, audit/EDITS docs, `funwake2/state/`, `baselines_g2.json`,
and `evaluator.py` (which encodes holdout/test roles) — is OUTSIDE the scope.
Containment is belt-and-suspenders: (1) the engine runs with **cwd** = the scope
(Gemini `subprocess(cwd=)`, Claude `ClaudeAgentOptions(cwd=)`), and (2) the Claude
engine ships `allowed_tools=[]` (no file tools at all). `sanitize()` strips seed
docstrings + redacts residual forbidden tokens (e.g. `native.py`'s `results/…`
provenance); `assert_clean()` scans the materialized scope and RAISES on any
forbidden path/holdout token, so a launch cannot proceed with a leaky workspace;
`scan_tree()` is reused post-run to grep the mutator transcript. Unit-tested:
`tests/test_workspace_scoping.py` (4 tests) — scope-contains-only-allowed,
assert_clean-raises-on-leak, raw-AEP-feedback-rejected, transcript-token-flagged.

---

## 5. DRY RUN — machinery validation (MOCK mutator, NO LLM spend)

MOCK mutator (`MockEngine`), `--fake-eval` (jax-free deterministic stand-in) for
the state-machine checks; the real-eval cache/skip-on-resume path is covered by the
unit suite (`funwake2/controller/tests/`, **13/13 pass**: machinery 8 + preflight 1
+ workspace-scoping 4). All scenarios via `funwake2/controller/run_dry.py`.

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

- **`parque_n30_uniform` 0/10 reference — RESOLVED (item 1 + 2).** Reconciled as a
  genuine unidirectional-rose × tight-zone × scale effect, NOT a v2 multizone
  fidelity bug (§1a: same-object init/penalty/SDF; lr0=50 reproduces the old 2/10
  regime; the "12/12" reference was best-of-50 multistart; "9–10/10" was N=20-DEI).
  **Swapped** to `parque_n14_uniform` (largest uniform-Parque N with a 10/10 c·D
  reference; §1a). `parque_n30_uniform` moved to the capability-frontier tier
  (§2b), informational-only. No open decision — flagged FYI; revert the swap only
  if you'd rather keep n30 in stage-B as an infeasible-reference (scale-constant)
  cell.
- **NEW — real-engine smoke is ENV-BLOCKED on BOTH engines (needs your action).**
  The Phase-3 workspace-scoping launch gate is BUILT and unit-tested (§4g), but the
  live 2-mutation smoke could not run here — neither engine is usable in this env:
  - **Claude:** `CLAUDE_CODE_OAUTH_TOKEN` unset + `claude_agent_sdk` not installed.
    Needs `pip install claude-agent-sdk` + `claude setup-token` (an **interactive**
    login only you can run — e.g. `! claude setup-token`). `ANTHROPIC_API_KEY` is
    correctly ABSENT (preflight would refuse it).
  - **Gemini:** the installed CLI returns `IneligibleTierError` ("client no longer
    supported for Gemini Code Assist for individuals" → migrate to Antigravity).
  **LAUNCH IS CLAUDE-FIRST (item 3):** Gemini is NOT required for Phase-3 launch.
  Its restoration — an updated CLI/auth or a metered-API engine on Google credits —
  is a **parallel, non-blocking** task; the `gemini_cli.py` wiring is unchanged and
  ready. *Archival note:* the v1 Gemini CLI tier (individual Gemini Code Assist) is
  now **deprecated** upstream, which is why the v1-era `gemini -p` path no longer
  authenticates. Once Claude auth lands I run the scoped 2-mutation Claude smoke +
  `scan_tree` the full transcript for forbidden paths/holdout values. The gate CODE
  is complete and tested; the Claude smoke is the one launch-blocking item-4 sub-task.
- **n200 classification (gbar):** run native@c·D 5 seeds on gbar to classify
  n200 as a stage-B+ elite cell (if feasible) or a capability-frontier test cell
  (if infeasible). The Mac 1-seed probe was infeasible but is not authoritative.
- **Parque heterogeneous test cell:** built locally as the item-6 pre-test step
  (no gbar needed) — see §7a.
- **gbar env standup:** stage-B+ / high-N confirmatory runs + the full Phase-3 run
  need the clean-room env on gbar (the remaining long pole).

### 7a. Parque heterogeneous TEST problem — BUILT (item 6, local)
`parqo/build_problem_hetero.py` → `parqo/problem_parqo_hetero.json` (596 KB). Unlike
`build_problem.py` (homogeneous site-average), it PRESERVES the per-cell, per-sector
WAsP maps at hub height (70 m): `Weibull_A`, `Weibull_k`, `Sector_frequency`,
`Speedup`, `Turning` as `(12 wd × 20 y × 20 x)` grids + the grid coords, so a
deployment-time heterogeneous evaluator (py_wake `ParqueFicticioSite`, natively
heterogeneous, or an interpolating wrapper) gives each turbine its local climate.
The resource is genuinely heterogeneous (Weibull_A 1.49–12.62 m/s, std 2.22; Speedup
0.18–1.68). It lives in the SOURCE tree (firewalled from mutators; a one-touch TEST
cell). The heterogeneous eval is wired at the deployment/test stage (not the Mac
evolution loop); no runnable CELLS entry is registered yet so it cannot be
accidentally scored during search.

---

## 8. Confirmation

- **NO Phase-3 launch.** No multi-generation real search; no real Claude/Gemini
  mutation was invoked (dry run = MockEngine only).
- Nothing under `runs/`, `archive/`, or `results/baselines*.json` was modified.
- Pre-registration updated + frozen: `results/funwake2_prereg/PREREGISTRATION.md`
  (per-cell G2 baselines + per-cell floors filled).
