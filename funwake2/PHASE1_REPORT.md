# FunWake-2 — Phase 1 report (build + incumbent-port validation)

**Scope: BUILD + SMOKE-TEST ONLY.** No evolutionary search, no mutators, no
OpenEvolve/ShinkaEvolve, no cascade/controller was created or run. STOP before
Phase 2. All work is additive under `funwake2/`; nothing under `runs/`,
`archive/`, or `results/baselines*.json` was modified (verified: `git status`
shows no changes to those paths).

Authoritative inputs: `specs/funwake2_spec.md` (incl. SIGN-OFF ADDENDUM) and
`results/funwake2_prereg/PREREGISTRATION.md`. Baseline targets from
`results/lr0_diameter_rule/aggregate.json`.

---

## Built files (all under `funwake2/`)

| file | purpose |
|---|---|
| `skeleton_v2.py` | Scale-aware skeleton. New signature `schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0)`. Computes **`alpha0 = mean(|∇_x J|,|∇_y J|)/D`** at the wind-aware init (D-2). No hardcoded lr. Single-polygon (DEI/ROWP) + multizone (Parque) dispatch; Adam loop verbatim from the vetted v1 skeletons. `total_steps` default = 8000 (D-1). |
| `seeds/native.py` | Native TopFarm monotone port. Internal `lr0 = (200/240)·D` (=0.8333·D; exactly 200 at D=240), constant lr first `total_steps//3`, compounding-product decay to **absolute** `gamma_min`, betas 0.1/0.2, `alpha(t)=alpha0·D/lr(t)` (recovers native `mean|∇J|/lr`, computed from **D** not the driver `/lr`). |
| `seeds/iter192.py`, `seeds/iter181.py`, `seeds/iter118.py`, `seeds/_port.py` | Incumbent ports. Thin wrappers calling the archived `schedule_fn` with **internal `lr0 = (50/240)·D` (=0.2083·D; exactly 50 at D=240)**, alpha0 passed through unchanged. |
| `evaluator.py` | Minimal scorer wrapping the v2 skeleton → AEP + feasibility on a (cell, seed, gamma_min). NOT the Phase-3 cascade/controller. |
| `gates/g3_bit_identity.py`, `g4_alpha0.py`, `g6_texture_floor.py`, `g7_gamma_min.py`, `aggregate_gates.py` | Gate runners. |
| `state/gates/*.json` | Per-(cell,seed) checkpointed results (69 files). |

Cells (from `lr0_diameter_rule`): **DEI n50** (`results/problem_dei_n50.json`,
D=240, ms=960, own rose), **ROWP n74** (`results/problem_rowp.json`, D=198,
ms=792, own rose), **Parque n20** (multizone `parqo/problem_parqo.json`, D=80,
ms=160, **DEI rose** from `results/matrix/problem_dei_n50_rosedei.json`).

---

## Gate results

### G1 — native fidelity, 6000-step (GO/NO-GO). **PASS (3/3 cells)**
Native port, `lr0=0.833·D`, `total_steps=6000`, `gamma_min=0.01`, 10 seeds
(0–9), feasibility-gated (DEI/ROWP: `bnd_pen<1e-3` & `min_dist≥0.99·ms`; Parque:
`max_zone_sdf≤0.1 m` & spacing).

| cell | baseline mean | native mean (10 seeds) | Δ | noise thr | feas | verdict |
|---|---|---|---|---|---|---|
| DEI n50 | 5561.34 | **5560.79** (std 2.55, sem 0.81) | **−0.55** | ±2.91 | 10/10 | **PASS** |
| ROWP n74 | 4261.72 | **4261.72** (std 2.61, sem 0.83) | **+0.00** | ±3.31 | 10/10 | **PASS** |
| Parque n20 | 231.06 | **231.24** (std 1.36, sem 0.43) | **+0.18** | ±1.50 | 10/10 | **PASS** |

Every cell reproduces the `lr0_diameter_rule` baseline within the noise floor;
feasibility ≥ baseline (Parque baseline was 9/10 @0.1 m — the port is 10/10).

**Reproduction note (important, expected).** The native port's lr trajectory is
**bit-identical** to `results/lr0_tuning/baseline_schedule.py` at lr0=200 (lr max
abs diff = 0.0, same bisection `mid`). Per-seed AEP differs by ~1–2 GWh (within
the seed std 2.06) because the mandated **`/D` alpha0** (D-2) makes the schedule
form `alpha0·D` *inside* the JIT trace, which XLA fuses/re-rounds ~1 ULP
differently from the baseline's `alpha0·lr0`; that ULP difference amplifies in
the chaotic low-lr decay tail. The **10-seed mean** is faithful to ≤0.55 GWh on
all cells. This is why G1 is a within-noise gate (only G3 requires bit-identity).
On-machine determinism confirmed: the recorded pipeline reproduces DEI seed-0 =
5561.17 exactly here.

### G2 — 8000-step bridge → **IN-SEARCH c·D baseline** (flagged)
Native port, `total_steps=8000`, `gamma_min=0.01`, 10 seeds, all feasible.

| cell | mean AEP (GWh) | std | sem | feas |
|---|---|---|---|---|
| **DEI n50** | **5560.32** | 1.81 | 0.57 | 10/10 |
| **ROWP n74** | **4263.54** | 1.06 | 0.34 | 10/10 |
| **Parque n20** | **231.57** | 1.17 | 0.37 | 10/10 |

**These three numbers are the in-search c·D baseline** (fitness %-over-baseline
is measured against them, paired by seed).

### G3 — incumbent bit-identity at D=240 (BLOCKING). **PASS** — max diff EXACTLY 0
For each incumbent, `iterXXX_port(step,8000,D=240,ms,N,γ,alpha0)` vs archived
`iter_XXX(step,8000,lr0=50,alpha0)`, over 4 outputs × 4 fixed alpha0 values ×
8000 steps:

| incumbent | max abs diff | verdict |
|---|---|---|
| iter192 | **0.000e+00** | PASS |
| iter181 | **0.000e+00** | PASS |
| iter118 | **0.000e+00** | PASS |

Peak-lr/D check: `iter192_port` internal lr0 = **50.0** at DEI (D=240) → peak_lr
200.0 → **peak_lr/D = 0.8333**; at ROWP (D=198) internal lr0 = **41.25** → peak
165.0 → **peak_lr/D = 0.8333**. (The port ramps ~4× to exactly the 0.833·D scale
on every farm.)

### G4 — alpha0 normalization. **PASS** — driver `/lr` = 1.2× the shipped `/D`
DEI n50, seed 0: `mean|∇J| = 1.272405e-02`.
- `alpha0` (skeleton, `/D`)            = **5.301688e-05**
- `alpha0` (driver default, `/lr0=200`) = 6.362026e-05
- **ratio driver/skeleton = 1.200000** (= 1/0.833). Shipping `/lr` would be 1.2×
  off. The native port consumes the `/D` alpha0 (`alpha=alpha0·D/lr`), never the
  `/lr` form.

### G5 — feasibility smoke (gamma_min=0.01). **PASS** — all feasible
| schedule | DEI n50 | ROWP n74 | Parque n20 |
|---|---|---|---|
| native (10 seeds, from G2) | 10/10, 5560.32 | 10/10, 4263.54 | 10/10, 231.57 |
| iter192_port (3 seeds, 8000) | 3/3, 5559.60 | 3/3, 4264.45 | 3/3, 230.03 |

### G6 — Parque per-cell texture floor. **DONE**
Optimized Parque native layout (AEP 231.05 GWh) held fixed, turbine positions
jittered at mm scale (40 draws each), AEP re-scored:

| jitter σ | AEP std (GWh) | max abs dev (GWh) |
|---|---|---|
| 1 mm | **7.84e-03** | 3.60e-02 |
| 10 mm | **1.11e-01** | 2.44e-01 |
| 100 mm | 1.17e-01 | 4.27e-01 |

**Parque per-cell texture floor ≈ 0.1 GWh** (10 mm scale; ~0.008 GWh at 1 mm).
In relative terms 0.11/231 ≈ 0.048% vs DEI's 0.3/5500 ≈ 0.0055% — the Parque
floor is ~0.1 GWh **absolute** but ~9× stricter relatively, confirming the
prereg's concern that a single 0.3 GWh floor is wrong at Parque's scale. Feeds
the prereg per-cell floors (DEI ~0.3 GWh, **Parque ~0.1 GWh**).

### G7 — gamma_min responsiveness. **PASS** — schedule responds to the tolerance
Native port, gamma_min = 0.01 vs 1.0:

- **Terminal lr** (last step, DEI D=240): γ=0.01 → **0.01004 m**; γ=1.0 →
  **1.00199 m** (100× — the schedule decays lr to the supplied tolerance).
- **DEI n50** (3 seeds): γ=0.01 → **3/3 feasible**; γ=1.0 → **1/3 feasible**.
- **Parque n20** (3 seeds): γ=0.01 → **3/3 feasible**; γ=1.0 → **0/3 feasible**.

Feasibility collapses at the looser tolerance (the endgame lr stays too high to
snap turbines onto the boundary) — the schedule is demonstrably **not** invariant
to gamma_min.

---

## Summary

| gate | result |
|---|---|
| G1 native fidelity 6000-step (GO/NO-GO) | **PASS** 3/3 cells |
| G2 8000-step bridge (in-search baseline) | DEI 5560.32 / ROWP 4263.54 / Parque 231.57 |
| G3 incumbent bit-identity (BLOCKING) | **PASS** — max diff 0.000e+00 all three |
| G4 alpha0 1.2× | **PASS** — ratio 1.200000 |
| G5 feasibility smoke | **PASS** — native 10/10, iter192 3/3, all cells |
| G6 Parque texture floor | ≈0.1 GWh (10 mm), 0.008 GWh (1 mm) |
| G7 gamma_min responsiveness | **PASS** — terminal lr 0.01↔1.00 m; feas 3/3→0–1/3 |

**All gates pass / are computed. The go/no-go gate (G1) and the blocking gate
(G3) both pass. Phase 1 is complete; search is authorized to proceed in Phase 2,
but NO search/mutator/controller was built or run here — STOP.**

No search, mutator, OpenEvolve/ShinkaEvolve, or cascade/controller was created or
run. Nothing under `runs/`, `archive/`, or `results/baselines*.json` was touched.

---

## Sign-off riders (Phase-1 accepted; R1 resolved)

**R1 (G8, determinism) — RESOLVED at source.** The 1–2 GWh cross-build scatter
was NOT process RNG: same-process is exactly deterministic and, in-env, alpha0 +
AEP are bit-identical across fresh processes. Root cause = extreme alpha0
sensitivity (a 5th-sig-fig alpha0 change moves AEP ~1.5 GWh; pinned-literal test
confirmed), which explodes a sub-ULP cross-build alpha0 difference. **Fix
(shipped):** `skeleton_v2` round-trips alpha0 through float32 at the boundary →
canonical value all environments agree on. **G8 PASS:** native dei_n50 seed0 =
5560.7575185032 bit-identical across two fresh processes (diag + evaluator
paths). Floors NOT inflated; content-addressed resume safe; no forced in-process
pairing. Canonicalization shifts baselines slightly (5562.96→5560.76 on DEI
seed0, landing on the G1 mean 5560.79) — **G2 in-search baseline re-measured
post-fix** in Phase 2.

**R2 (G3 scope).** G3 is a **function-level transform test at matched alpha0**
(the port substitutes lr0→0.2083·D with the algebra unchanged; bit-identical at
D=240). End-to-end, the ported ancestors run under the `/D` alpha0 at
**~0.21–0.25× their historical penalty scale** (their historical alpha0 was
`/lr0=/50`; now `/D=/240` → ×50/240≈0.208), which G5 confirms is still viable.
**Gen-0 stage-B scores under the new regime are the ancestors' reference —
never their historical AEP numbers.**

**R3 (G7 scope).** G7 responsiveness was demonstrated **on the native seed**. The
ported incumbents **ignore `gamma_min` by construction** (their historical
schedule_fn predates the tolerance input), so the stage-C responsiveness check
**correctly bars an unmodified ancestor clone from deployment** — this is
intended selection pressure toward tolerance-aware schedules, not a defect.

**R4 (archive + prereg).** iter118's descriptor is **peak_lr/D = 1.354** (6.5×
of its internal 0.2083·D) → **>1.2 bin**; iter192 = 0.833 (0.8–1.2 bin); native's
peak = 0.833 but its coupling/terminal differ → the three seeded ancestors occupy
**three distinct archive cells at gen 0**. The G2 bridge per-cell baseline numbers
(re-measured post-G8) are published in the pre-registration as the frozen
in-search baseline.
