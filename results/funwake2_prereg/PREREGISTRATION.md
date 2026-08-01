# FunWake-2 — pre-registration (DRAFT, Phase-0)

Author sign-off artifact. SOURCE repo only — NEVER in a mutator workspace, so
the evolutionary search cannot see the deployment/test design. Frozen and
committed BEFORE Phase 3 (launch). Only numeric slots {…} in the selected
outcome may be filled post-hoc; wording, criterion, and test set are frozen.

Companion design: `specs/funwake2_spec.md`. Baseline = the c·D-tuned native
TopFarm-SGD schedule (lr scale 0.833·D, decay to gamma_min), same object the v1
diameter-rule analysis measured.

## Question
Under a scale-aware parameterization (no free lr0; exploration scale built from
rotor diameter D; gamma_min = the user's metre-valued constraint tolerance), can
LLM-driven evolutionary search discover a **deployable** TopFarm-SGD schedule
that measurably beats a properly-tuned (c·D) baseline on an unseen, pre-committed
test set — and survive real-TopFarm stochastic re-validation?

## Frozen protocol
- **Fidelity gate (Phase-1, must pass before search):** the native port
  (lr scale 0.833·D → gamma_min) reproduces `results/lr0_diameter_rule/` baseline
  scores within the noise floor on all shared cells. No reproduction ⇒ no search.
- **Search:** islands + MAP-elites evolutionary run (chassis per spec §3),
  cascade evaluator (spec §3.2), Claude Agent-SDK + Gemini-CLI mutators. Fitness
  = paired multi-seed mean AEP as %-over-c·D-baseline per training cell, hard
  feasibility gate at gamma_min, cross-cell aggregate per spec §2.1. Training
  cells span ≥2 rotor diameters incl. hard cells (spec §2.2). Holdout/test AEP
  firewalled from every mutator context/transcript (key-only).
- **Selection (holdout = ROWP, margin-aware, NEVER single-eval argmax):**
  deployed schedule = the elite maximizing ROWP validation-mean margin over the
  c·D baseline (≥5 seeds), feasibility-gated, margin > 0.3 GWh. If none clears
  the margin, deploy the native seed and record "no improvement."
- **TEST set (FROZEN — touched exactly once, at the deployment decision;
  D-7 resolved Phase-2):**
  1. **ROWP high-N** — `rowp_n200_roserowp` and `rowp_n300_roserowp`.
  2. **Parque real heterogeneous wind resource** — the currently-unused per-cell
     WAsP Weibull A/k + speedup/turning maps (`problem_parqo.json` uses the
     homogeneous site-averaged surrogate; the heterogeneous problem JSON is a
     pre-test build from `parqo/build_problem.py`).
  3. **Unidirectional extreme on an unseen farm** — `rowp_n74_uniform` (ROWP
     polygon + uniform/unidirectional rose).
  (If the gbar 5-seed classification finds DEI n200 infeasible, it joins the
  test set as the capability-frontier cell.) AEP is never observed before the
  one-time paired 30-seed test. High-N confirmatory cells are ONE-SHOT 30-seed
  runs where ~5-min evals are affordable (the evolution-loop ≤3-min limit does
  not apply to the test set).

## Deployment criterion (frozen — the confirmatory test)
On the pre-registered TEST set, the deployed schedule vs the c·D-tuned native
baseline under the **paired 30-seed protocol** (identical per-seed init layouts,
verified max coord diff = 0; per-seed Δ = AEP(deployed) − AEP(baseline); paired
mean, 95 % t-CI, Wilcoxon signed-rank; over seeds feasible in both arms).
**Deployable-improvement is claimed iff, on the test set: paired 95 % CI excludes
0 AND |mean Δ| > 0.3 GWh AND strict feasibility at gamma_min ≥ baseline** — AND
the **deployment-fidelity stage** passes: the winner re-validated in **real
TopFarm / py_wake with stochastic wind sampling** (via `ex_dei.py` ported to each
test farm) retains the improvement and feasibility under the true stochastic
optimizer. A pixwake-surrogate win that does not survive real-TopFarm stochastic
re-validation is recorded as NOT deployable.

## OUTCOME — SUCCESS  [insert verbatim if the criterion is met; fill {slots} only]
> Under a scale-aware parameterization with no free learning rate, LLM-driven
> evolutionary program search discovered a TopFarm-SGD schedule that improves on
> a properly-tuned (rotor-diameter-scaled) baseline. On a pre-registered,
> once-touched test set, the deployed schedule exceeds the c·D-tuned native
> baseline under a paired 30-seed comparison by {+Δ} GWh ({+Δ%}, 95 % CI
> [{lo},{hi}], p={p}), above the 0.3 GWh texture floor, at feasibility ≥ the
> baseline at the specified constraint tolerance, and the improvement survived
> re-validation in real TopFarm/py_wake with stochastic sampling ({fidelity Δ}).
> The behavioral signature of the discovered schedule — {peak_lr/D, terminal lr,
> alpha-coupling class, restart count} — and its full lineage back to the seeded
> ancestors are recorded, so the discovery is auditable against provenance.

## OUTCOME — NULL  [insert verbatim if the criterion is NOT met; fill {slots} only]
> Under a scale-aware parameterization with no free learning rate, LLM-driven
> evolutionary program search did not produce a deployable improvement over a
> properly-tuned (rotor-diameter-scaled) TopFarm-SGD baseline. On a
> pre-registered, once-touched test set, the deployed schedule's difference from
> the c·D-tuned native baseline under a paired 30-seed comparison is {Δ} GWh
> (95 % CI [{lo},{hi}], {spans zero / |Δ| < 0.3 GWh / lost feasibility / failed
> real-TopFarm re-validation}) — statistical parity or non-deployable. Against a
> fair baseline, the search's demonstrated contribution is the recovery and
> archiving of a scale-aware schedule family that matches expert-tuned SGD, with
> full lineage provenance, rather than a performance gain over it.

## Provenance requirements (for any "the system discovered X" claim)
- Every candidate: content hash, parent ID(s), mutation engine + model string,
  token/cost, behavioral descriptors, per-cell fitness, generation, timestamp.
- Seeded ancestors (native, iter_192/181/118 ports) logged as generation-0 with
  their port transforms — so any claimed novelty is measured against what was
  seeded, and "discovered" is distinguishable from "recombined a seed."

## Post-run record (data only, no reframing)
- Deployed schedule + lineage path + selection margin: {…}
- Test-set paired table + fidelity-stage result: {…}
- Selected outcome + criterion components (CI, |Δ|, feasibility, fidelity): {…}
- Total mutations, tokens, cost, model strings; abort-rule status: {…}

---

# SIGN-OFF ADDITIONS (Phase-0 approved) — before the criterion is frozen

- **Per-cell texture floors (BLOCKING before freeze).** The 0.3 GWh floor is
  DEI-calibrated (~0.005% of ~5500 GWh); at Parque's ~230 GWh scale it is ~25×
  stricter in relative terms. Phase-1 must **re-measure the evaluation-texture
  floor on ≥1 Parque cell** (the mm-scale wake-cone-mask texture, in absolute
  GWh at that farm's scale) and set **per-cell |mean Δ| floors** accordingly.
  The deployment criterion uses the per-cell floor for each test cell, not a
  single 0.3 GWh. Freeze the per-cell floors here after measurement.
  **MEASURED (Phase-1 G6):** DEI ≈ **0.3 GWh** (prior es_mechanism calibration);
  Parque ≈ **0.1 GWh** absolute (AEP std 0.11 GWh @10 mm layout perturbation) —
  ~9× stricter relatively than DEI. **ROWP ≈ 0.64 GWh** (MEASURED Phase-2 G6-ROWP,
  `funwake2/gates/g6_rowp_floor.py`, rowp_n74: 10 mm-scale AEP std on the 4258.73
  GWh optimized native layout; 1 mm → 0.23, 100 mm → 1.00 GWh; ~0.015% of AEP —
  ~3× relatively noisier than DEI).
  **Per-cell deployment floors FROZEN: DEI = 0.3, Parque = 0.1, ROWP = 0.64 GWh.**
  The deployment criterion uses the ROWP floor (0.64) for the ROWP test cells
  (`rowp_n200`, `rowp_n300`, `rowp_n74_uniform`) and the Parque floor (0.1) for the
  Parque heterogeneous test cell — never a single 0.3 GWh.
  RESOLVED at source (G8, R1). Diagnosis: same-process is exactly deterministic;
  across fresh processes alpha0 AND AEP are bit-identical in-env — the cross-BUILD
  1–2 GWh gap was a sub-ULP alpha0 difference exploded by extreme sensitivity (a
  5th-sig-fig alpha0 change moves AEP ~1.5 GWh; pinned-literal test confirmed).
  **Fix (shipped in skeleton_v2):** alpha0 is round-tripped through float32 at the
  skeleton boundary → a canonical value every environment agrees on (float64
  reductions agree to ~14 sig figs >> float32's 7). **G8 PASS:** same
  (schedule, cell, seed) matches to 0.0000 across two fresh processes.
  Floors are therefore NOT inflated by process noise; the 30-seed paired test
  keeps full power at ~0.3 GWh margins. (Canonicalization shifts the baseline
  AEPs slightly — the frozen in-search G2 baseline is RE-MEASURED post-fix.)
- **gamma_min responsiveness (stage-C, D-7).** Elites must demonstrably respond
  to the tolerance input: re-score at **gamma_min = 1.0 m** and confirm the
  schedule's behavior (and feasibility/terminal-lr) changes vs gamma_min = 0.01 m.
  A schedule invariant to gamma_min is rejected regardless of AEP — it is not a
  faithful TopFarm schedule.
- **Test set (frozen candidates):** ROWP high-N (n200/n300), Parque real
  heterogeneous wind resource (currently unused), unidirectional extreme on an
  unseen farm. Final composition + per-cell floors frozen at end of Phase-1.

---

# n200 / FITNESS ADDITIONS (Phase-2) — before criterion freeze

- **Fitness normalization with infeasible reference (blocking, all cells).**
  `score_c = 100·(AEP_cand − AEP_ref,c)/AEP_ref,c` treats `AEP_ref,c` as a scale
  constant valid even if the c·D reference is infeasible at cell c; the
  candidate's feasibility at gamma_min is the independent HARD gate. Cells with an
  infeasible reference still normalize candidate AEP but confer no feasibility
  credit.
- **Capability-frontier cell (n200, pending gbar classification).** native@c·D at
  n200 is classified on gbar (5 seeds). If infeasible, n200 is the
  capability-frontier cell: a candidate reaching STRICT feasibility at n200 is a
  qualitatively new result and **enters this confirmatory test set** (added to the
  frozen test-set list upon classification). If feasible, n200 is an ordinary
  stage-B+ elite cell and NOT a test cell.
- **Test-set eval budget (point 7).** Confirmatory high-N cells are ONE-SHOT
  30-seed runs; ~5-min/eval is affordable there. The ≤3-min evolution-loop cost
  limit applies ONLY to per-generation stage-B, never to this confirmatory test.
- **Stage-B high-N cell frozen = DEI n120/dei** (159 s/eval, feasible reference).
  n200 is stage-B+ elite (gbar-only), not per-generation stage-B.

---

# FROZEN in-search c·D baseline (Phase-2, post-G8) — the number fitness beats

Native c·D schedule (lr scale 0.833·D → gamma_min), `total_steps=8000`,
`gamma_min=0.01`, **10 seeds (0–9)**, feasibility-gated, canonicalized alpha0 (G8
float32). Source: `funwake2/controller/baselines_g2.json`
(`complete_10seed_all_cells=true`; per-seed AEP retained for paired scoring).
`score_c = 100·(AEP_cand − AEP_ref,c)/AEP_ref,c`, paired by seed.

| cell | N | D | mean AEP (GWh) | std | feasible |
|---|---|---|---|---|---|
| `dei_n50` | 50 | 240 | 5560.393 | 1.462 | 10/10 |
| `dei_n80_omnidir` | 80 | 240 | 8818.125 | 3.719 | 10/10 |
| `dei_n120_rosedei` | 120 | 240 | 13029.988 | 7.496 | 10/10 |
| `dei_n50_uniform` | 50 | 240 | 5598.413 | 1.315 | 10/10 |
| `parque_n20` | 20 | 80 | 231.505 | 0.688 | 10/10 |
| `parque_n30_uniform` | 30 | 80 | 356.171 | 4.884 | **0/10 (infeasible ref)** |
| `parque_n10_omnidir` | 10 | 80 | 127.002 | 0.319 | 10/10 |

`parque_n30_uniform`'s c·D reference is genuinely infeasible (`max_sdf` 3–37 m; 30
turbines do not pack into the Parque zones under a unidirectional rose). Per the
scale-constant fitness patch its `AEP_ref` normalizes candidate AEP but confers no
feasibility credit; candidate feasibility at gamma_min is the independent hard
gate. Author decision at sign-off (PHASE2_REPORT §7): retain as a hard stage-B cell
or swap for a feasible lower-N Parque unidirectional cell — recorded here so the
frozen protocol reflects whichever is chosen before launch.
