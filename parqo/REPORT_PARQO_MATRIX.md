# ParqueFicticio generalization matrix

Third held-out site (Criado Risco et al. 2024): Vestas V80 (D=80 m, a third
turbine model), **five disconnected irregular inclusion zones**, 2 D min
spacing. Multizone SDF skeleton. Tests whether the deployed schedules transfer
to qualitatively different topology (disconnected zones — impossible on the
single-polygon matrix).

## Design
- 3 single-run schedules: baseline seed, Gemini iter_192, Claude dual-bump (iter_192)
- Plus a **best-of-50 multistart baseline** (seed schedule × 50 init seeds/cell)
- 6 counts {10,15,20,25,30,35} × 4 roses {dei,rowp,omnidir,uniform} = 24 cells/schedule
- Deterministic full-rose AEP + zone-containment + spacing feasibility

## Headline: the advantage is constraint precision, not AEP — and it is tolerance-dependent

Feasible cells / wins out of 24 (best-of-50 multistart baseline vs single-run schedules):

| tol   | MS-base feas/wins | Gemini feas/wins | Claude feas/wins |
|-------|-------------------|------------------|------------------|
| 0.1 m | 7 / 6             | 21 / 2           | **23 / 16**      |
| 0.5 m | 18 / 16           | 22 / 1           | 23 / 7           |
| 1.0 m | **23 / 23**       | 23 / 1           | 23 / 0           |
| 2.0 m | 23 / 23           | 23 / 1           | 23 / 0           |
| 5.0 m | 24 / 24           | 24 / 0           | 23 / 0           |

- **Strict (0.1 m)**: deployed dual-bump dominates (16/24 wins). The multistart
  baseline's **median feasible-start rate is 0/50** — best-of-50 restarts
  essentially never lands strictly inside all five disconnected zones. The
  LLM-discovered schedule achieves strict feasibility DETERMINISTICALLY in one
  run.
- **Loose (≥1 m)**: the multistart baseline sweeps (23–24 wins). Its
  ~1 m-near-feasible layouts, once admitted, carry higher AEP than the
  deployed schedules everywhere. The baseline was not wasting effort on strict
  containment; the deployed schedules were.

## Interpretation
On hard constrained geometry (disconnected zones), the value of the
LLM-discovered schedule is **constraint precision**: it hits engineering-exact
feasibility in a single deterministic run where 50 multistart restarts cannot.
Relax the containment constraint by one meter and the advantage inverts — the
baseline's higher-AEP near-feasible layouts win. This is a sharper claim than
"the schedule makes more energy": it localizes the transfer benefit to
feasibility, not objective.

## Files
- `parqo/parqo_matrix.json` — 72 single-run cells (baseline/gemini/claude × 24)
- `parqo/parqo_baseline_ms.json` — 24 cells, 50 starts each, per-seed metrics
  (any tolerance re-gateable post-hoc)
- `parqo/run_parqo_matrix.py`, `parqo/run_parqo_baseline_ms.py` — runners
- Site build + multizone skeleton: `parqo/build_problem.py`,
  `parqo/skeleton_multizone.py`; zones digitized in `parqo/inclusion_zones.json`

## Caveats
- Homogeneous site-averaged climate (pixwake takes one ambient per flow case),
  so AEP is NOT comparable to the paper's heterogeneous-flow ~100–116 GWh.
- Zones digitized from the paper's Fig. 5 raster (~6 m simplification), not
  published machine-readable polygons.
