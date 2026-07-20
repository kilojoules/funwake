# pixwake ↔ py_wake equivalence — internal validation

**Purpose:** justify treating pixwake (the JAX-fast fork used as FunWake's
compute engine) as numerically interchangeable with py_wake for the AEP and
regret quantities our papers depend on.

**Status:** gate PASSED. py_wake stays out of paper text; pixwake is engine.

## 1a — Existing-suite summary

`dependencies/pixwake/tests/` contains zero pywake-equivalence tests. Only:
- `test_boundary.py` — polygon SDF/penalty correctness
- `test_greedy_grid.py` — GreedyGridSearch smoke
- `test_ift_gradients.py` — implicit-function-theorem gradient checks

One-off script `dependencies/pixwake/scripts/verify_A0.02_pywake.py` exercises
TurboGaussianDeficit against py_wake but asserts no tolerance and is not
collected by pytest.

**Asserted quantity / tolerance: none.** The validation below fills that gap.

## 1b — Published-layout AEP under both engines

Setup: IEA Wind 740-10-MW ROWP **irregular** layout (74 × IEA 10 MW
turbines, Borssele zone). Resource: 12-sector Weibull(A, k) at hub
height + 1° wd × 1 m/s ws grid, NOJ Jensen wake k = 0.05, SquaredSum
(RSS) superposition, RotorCenter rotor-averaging, no turbulence model.
Identical (x, y), turbine yaml, and joint probabilities used in both
engines (see `run_pywake_published.py` → dumps setup → `run_pixwake_mirror.py`).

| Engine            | Irregular AEP (GWh) | Regular AEP (GWh) |
|-------------------|--------------------:|------------------:|
| py_wake 2.6.20    | 3426.902698         | 3381.934211       |
| pixwake (master)  | 3426.902695         | 3381.934208       |
| **Δ engine**      | **−3.25 × 10⁻⁶**    | **−3.20 × 10⁻⁶**  |
| **Δ engine (%)**  | **−9.5 × 10⁻⁸**     | **−9.5 × 10⁻⁸**   |

Both engines agree to float64 round-off (~10 µGWh on a ~3500 GWh signal).

**Published reference (740-10 report): 3429.63 GWh** for Irregular.
Both engines reproduce **3426.90** — same number to round-off, **−2.73 GWh
(−0.080 %) vs published**. The 0.08 % gap is **py_wake version drift**
between whatever py_wake generated the original report and 2.6.20, not an
engine gap. Same drift would appear if we re-ran the official
`Examples/pywake_ex.py` from the cloned 740-10 repo with current py_wake;
we did and it gives 3426.90.

Conclusion 1b: pixwake reproduces py_wake's published-config AEP to
numerical noise; py_wake (current) reproduces the published number to
0.08 %. Engine-level equivalence ✓.

## 1c — Cancellation-sensitive regret check

Pair: Irregular (3426.902695 GWh) vs Regular (3381.934208 GWh) — same
Weibull resource, same wake. Treat this as a regret cell — exactly the
kind of large-AEP-minus-large-AEP that catastrophic cancellation worries
about.

| Quantity                | py_wake (GWh) | pixwake (GWh) |
|-------------------------|--------------:|--------------:|
| AEP Irregular           | 3426.902698   | 3426.902695   |
| AEP Regular             | 3381.934211   | 3381.934208   |
| **Regret = ΔAEP**       | **44.968487** | **44.968487** |
| **Δ engine on regret**  |               | **≈ 5 × 10⁻⁸ GWh** |

The per-AEP errors are systematic (same sign, same magnitude) and
**partially cancel** when subtracting — so the engine contribution to the
regret signal is even *smaller* than the per-eval absolute error.

## 1d — Bound vs K = 500 multistart noise floor

- Per-eval engine error (worst observed): **3.25 × 10⁻⁶ GWh**
- Triangle-inequality bound on regret error: 2 × per-eval = **6.5 × 10⁻⁶ GWh**
- K = 500 multistart noise floor (per `paper_schedules/main.tex`,
  conservative regret stagnation interval): **~20–50 GWh**
- Engine contribution / noise floor: **≤ 3 × 10⁻⁷** = **~7 orders of
  magnitude below** noise floor

For any AEP-difference signal we expect to report (6 – 100 GWh range),
the engine is irrelevant.

## Gate decision

**PASS.** pixwake ↔ py_wake difference on the cancellation-sensitive
quantity is ≤ 5 × 10⁻⁸ GWh, vs a ~20 GWh multistart noise floor. We will
not discuss py_wake in the paper. pixwake stays the engine.

## Artifacts

- Script: `validation/pixwake_pywake/run_pywake_published.py`
- Script: `validation/pixwake_pywake/run_pixwake_mirror.py`
- Results: `validation/pixwake_pywake/pywake_{irregular,regular}.json`
- Results: `validation/pixwake_pywake/pixwake_{irregular,regular}.json`
- Setup dumps (joint probabilities, layout, turbine curves):
  `validation/pixwake_pywake/pywake_{irregular,regular}_setup.json`
- Source data: `/Users/julianquick/portfolio_copy/IEA-Wind-740-10-ROWP/`
  (commit hash recorded by `git -C ../IEA-Wind-740-10-ROWP rev-parse HEAD`)
- py_wake version: 2.6.20
- Environment changes (in `pixi.toml`): added `hdf5`, `netcdf4` conda
  deps; added pypi deps `py_wake>=2.6.20,<3` and `windIO>=2.1.1,<3`.

## Caveats

1. **Pixwake has no native Weibull-marginalized AEP path.** The mirror
   script imports the joint-probability matrix from py_wake's setup
   dump. This is the right thing to do for an engine-equivalence test
   (eliminates wind-rose-interp drift) but means a pixwake-only Weibull
   AEP integrator is still missing — that gap is the subject of Part 2
   of the parent plan (restore stochastic AEP gradients with
   sector-Weibull sampling).
2. Both engines use SquaredSum superposition + NOJDeficit(k=0.05) +
   RotorCenter rotor-averaging + no turbulence model — matching the
   published 740-10 example. Other model combinations are not tested
   here.
3. The engine-equivalence tolerance does NOT cover gradients. The
   skeleton.py optimizer uses `jax.grad` through pixwake's sim.
   `test_ift_gradients.py` covers IFT-based gradients but not the
   per-eval-AEP gradient path used by FunWake. Flagged for follow-up.
