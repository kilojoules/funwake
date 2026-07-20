# parqo — ParqueFicticio inclusion-zone case study for pixwake

Re-implementation of the ParqueFicticio case study from Criado Risco et
al. (2024), *Gradient-based wind farm layout optimization with inclusion
and exclusion zones*, WES 9, 585-600 (doi:10.5194/wes-9-585-2024):
**12 x Vestas V80-2.0** (D = 80 m, hub 70 m), **min spacing 2 D = 160 m**,
**five irregular disconnected inclusion-zone polygons**, optimized with
the deployed dual-bump schedule (`runs/schedule_only_5hr/
iter_192.py`, the schedule used for `paper/gen_deployed_layouts.py`).

## Inclusion zones

No machine-readable polygon coordinates are published (the paper's repos
carry only the site rasters), so the five zones were digitized from the
vector-accurate raster of Fig. 5: thick inclusion-zone strokes segmented
by erosion, axis calibration from tick marks (~1.4 m/px), stroke
centerlines traced and simplified to ~6 m. Stored with provenance in
`inclusion_zones.json` (areas 0.291 / 0.236 / 0.144 / 0.036 / 0.008 km²).

## Approximations vs the paper

- pixwake's `WakeSimulation` takes one ambient wind per flow case, so
  the heterogeneous WAsP maps (per-cell Weibull A/k, speedup, turning)
  are reduced to a homogeneous site-averaged climate at hub height
  (12 sectors, speed = A·Γ(1 + 1/k)). AEP values are therefore NOT
  comparable to the paper's heterogeneous-flow numbers (~100-116 GWh).
- Optimizer is the funwake fixed Adam skeleton + dual-bump schedule
  (8000 steps), not SLSQP/relaxation/smart-start.

## Files

- `inclusion_zones.json` — digitized zone polygons (UTM meters)
- `build_problem.py` — site climate + V80 + zones → `problem_parqo.json`
- `schedule_dual_bump.py` — verbatim copy of iter_192 (deployed schedule)
- `skeleton_multizone.py` — fixed Adam skeleton with multi-polygon
  boundary: penalty = sum(relu(min-over-zones SDF)²), area-weighted
  candidate-grid init inside zones
- `run_parqo.py` — runs the skeleton with the dual-bump schedule, scores
  AEP + feasibility, writes `layout_parqo.json` / `results_parqo.json`
- `plot_layout.py` — zones + turbines → `parqo_layout.png`

## Usage

```bash
pixi run python parqo/build_problem.py
pixi run python parqo/run_parqo.py [--seed N]
pixi run python parqo/plot_layout.py
```

## Results (homogeneous surrogate)

| seed | AEP (GWh) | min spacing (m) | feasible |
|------|-----------|-----------------|----------|
| 0    | 26.192    | 237.8           | yes |
| 1    | 26.047    | 242.8           | yes |
| 2    | 25.815    | 216.4           | yes |

All 12 turbines end inside inclusion zones (max zone SDF <= 0), with the
allocation roughly area-weighted: 4 in the SE zone, 4 in the north zone,
2 in the SW strip, 1 in the NW blob, 1 in the tiny central sliver.
