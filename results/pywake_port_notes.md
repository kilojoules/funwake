# py_wake port of the unit-test path

Date: 2026-07-23. Scope: the UNIT TEST path only
(`playground/test_optimizer.py` + `tools/run_tests.py`). The
scoring/benchmark stack stays on the vendored pixwake for now.

## What changed

New (all license-clean, no pixwake imports):

- `playground/penalties_np.py` — plain-numpy TopFarm penalty
  formulations (boundary: sum of squared outside-distances to a convex
  CCW polygon; spacing: sum over violated pairs of
  `min_spacing^2 - d^2`) plus analytic gradients.
- `playground/pywake_adapter.py` — `Curve` / `Turbine` /
  `WakeSimulation` / `BastankhahGaussianDeficit` mirroring the small
  simulation interface the tests use, backed by py_wake 2.6.20.
  Also provides `make_neg_aep()` returning the negative-AEP objective
  [GWh] and its (x, y) gradient via py_wake's autograd backend.
- `playground/skeleton_pywake.py` — port of `skeleton.py` with the
  identical `run_with_schedule` contract: same wind-aware grid init
  (identical `jax.random` seeding, so initial layouts are bitwise
  equal), lr0=50.0, `alpha0 = mean|initial AEP grad| / lr0`, per-step
  `schedule_fn(step, total_steps, lr0, alpha0) -> (lr, alpha, b1, b2)`,
  Adam on `grad_aep + alpha * grad_constraint`, optional early-stopping
  rule. The JAX fori_loop is replaced by a plain-numpy Adam loop; the
  schedule is precomputed once (it depends only on step/lr0/alpha0).

Modified:

- `playground/test_optimizer.py` — imports switched from pixwake to
  `pywake_adapter` + `penalties_np`; `skeleton` → `skeleton_pywake`;
  schedule runs use `TEST_TOTAL_STEPS = 5000` (was 8000, see below);
  `run_via_harness` no longer injects a hardcoded pixwake PYTHONPATH
  (it passes the caller's PYTHONPATH through — see "Remaining pixwake
  surface").
- `tools/run_tests.py` — no longer prepends `dependencies/pixwake/src`
  to PYTHONPATH; subprocess timeout 180 s → 600 s (the py_wake suite
  takes ~4–4.5 min).
- `.gitignore` — whitelists the three new playground modules.

`grep -rn pixwake playground/test_optimizer.py tools/run_tests.py
playground/skeleton_pywake.py playground/pywake_adapter.py
playground/penalties_np.py` → no matches.

## Model-config mapping (pixwake → py_wake)

| pixwake | py_wake |
|---|---|
| `WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))`, fixed-point solve | `PropagateDownwind(site, wt, wake_deficitModel=..., superpositionModel=SquaredSum())` (equivalent: wake influence is strictly downstream, so the fixed point equals downstream propagation) |
| `BastankhahGaussianDeficit(k=0.04)` (ceps=0.2, ctlim=0.899, ct2a_madsen, use_effective_ws=False) | `BastankhahGaussianDeficit(k=0.04)` — identical defaults and identical formulation (verified line-by-line: same beta/epsilon, same `min(1, 2*ct2a(ct*D^2/(8*sigma^2)))` centerline, same Madsen ct2a coefficients (0.2460, 0.0586, 0.0883)) |
| deficit masked to `dw > 0` AND `|cw| < 2*sigma` | `dw > 0` implicit in PropagateDownwind; the 2-sigma radius mask is NOT in py_wake — added via `_RadiusMaskedBastankhah` subclass in the adapter (`deficit * (cw^2 < wake_radius^2)`) |
| `SquaredSum`: `ws_eff = ws_amb - sqrt(sum d^2)` | `SquaredSum()` — same |
| ct from source turbine's EFFECTIVE ws | same in PropagateDownwind |
| `ti_amb=None` (no turbulence model) | `UniformSite(ti=0.1)`; TI value inert (constant-k deficit, no turbulence model) |
| `Turbine` curves via `jnp.interp` (separate ws grids allowed) | `PowerCtTabular(method='linear', additional_models=[])` on the union of the power/ct ws grids (piecewise-linear interp onto the union grid is exact, so both curves are reproduced exactly); fixture/problem speeds stay inside the table so end-clamping differences never trigger |
| AEP `sum(p_kw * weights) * 8760 / 1e6` GWh | identical formula on `wfm._run(...)[2]` power (W → kW) |
| gradients: JAX autodiff | py_wake `utils.gradients.autograd` |

## Parity numbers (pixwake vs port)

- Penalties (8 random 25-turbine layouts, points inside and outside):
  boundary worst rel err 1.4e-16, spacing 1.7e-16; analytic penalty
  gradients exactly equal to `jax.grad` of pixwake's (0.0 diff).
- AEP on fixed layouts: stressed-fixture 25-wt random layout
  41.861268 vs 41.861269 GWh (+2e-8); 25-wt aligned row, heavy wake,
  15.641176 vs 15.641176 (-3e-8); DEI turbine + 24-case rose
  2749.141483 vs 2749.141487 (+1.5e-9). All far below the 1 % gate.
- AEP gradient (3-wt interacting layout): max rel diff vs pixwake's
  jax.grad = 1.8e-10; py_wake autograd vs central finite differences
  = 2.7e-7 worst rel err (also 7-figure agreement on the 25-wt
  fixture).

No irreducible model-config difference remained after adding the
2-sigma mask.

## Performance and the step-count decision

- py_wake gradient cost: 62.9 ms/step (25 wt, 4 wind cases; forward
  alone 4.5 ms — the rest is autograd through PropagateDownwind's
  sequential loop), 3.6 ms/step (3 wt). pixwake/JAX is far faster
  (jitted fori_loop), so 8000 steps → ~9 min suite on py_wake.
- Decision: `TEST_TOTAL_STEPS = 5000`. Verified with pixwake at the
  SAME setting that the reference qualitative pattern holds
  (stressed_boundary FAIL, penalty 0.139 ≈ same order as the 8000-step
  0.079). Scan of pixwake seed_schedule stressed boundary penalty vs
  steps: 8000 → 0.079 (FAIL), 7000 → 0.151 (FAIL), 6000 → 0.068
  (FAIL), 5000 → 0.139 (FAIL), 4500 → 0.0048 (marginal FAIL),
  4000 → 0.000 (anomalous PASS). Do not lower below 5000.
- Ported suite wall time: 4 min 06 s (seed_schedule), 4 min 26 s
  (iter_192). `tools/run_tests.py` timeout raised to 600 s.

## End-to-end pattern

`pixi run python tools/run_tests.py results/seed_schedule.py --quick`
(ported, 5000 steps): 8/9 PASS with exactly one FAIL —
`stressed_boundary: penalty=0.038768 (need < 1e-3)`; spacing 600.1 m
PASS. This reproduces the pixwake reference pattern (8000 steps:
penalty 0.079338; pixwake at 5000 steps: 0.138993).

`runs/schedule_only_5hr/iter_192.py` (ported): 8/9 with
`stressed_boundary: penalty=0.005180` FAIL, while pixwake at 5000
steps passes (penalty 6.2e-5). This is a knife-edge threshold effect,
not model mismatch: the 1e-3 gate corresponds to ~3 cm outside the
boundary, and iter_192's final residual is chaotic at that scale.
Ported penalties across tiny perturbations: 5000 steps → 5.2e-3;
5000 steps with 1 µm initial offset → 1.2e-3; 5050 steps → 0.0
(PASS). pixwake at 4990/5010/5050 → 1.2e-4/0.0/6.9e-5. Both backends
leave mm-to-cm boundary residuals for iter_192; which side of the
threshold they land on is not reproducible across numeric backends.

## Remaining pixwake surface (outside the test path)

Still importing pixwake (untouched, for a future full migration):

- `playground/skeleton.py` (`pixwake.optim.sgd` penalties; used by the
  scoring stack, no longer by the tests)
- `playground/harness.py` (`pixwake` core + deficit) — consequence:
  the FULL test mode of `test_optimizer.py` (with a problem.json) runs
  the optimizer via this harness and now requires the caller to set
  `PYTHONPATH=dependencies/pixwake/src` externally; the `--quick` unit
  path needs nothing. The post-harness layout checks (boundary/
  spacing/AEP) in `check_layout` already use the py_wake adapter.
- `tools/run_optimizer.py`, `tools/test_generalization.py` (set
  pixwake PYTHONPATH; drive the pixwake scorer)
- `benchmarks/` (firewalled scorer)
- `dependencies/pixwake/` (the vendored library itself)
- `agent_cli.py` / docs reference the pixwake-based flow.
