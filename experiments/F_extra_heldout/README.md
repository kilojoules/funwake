# Experiment F — Additional held-out polygon

## Reviewer concern (implicit)
"Single training farm (N=1 held-out polygon)" appears in the paper's
own limitations. A reviewer can ask: are the schedules robust across
*polygons*, or did the LLM happen to find something that works on the
two specific shapes you tested?

## Hypothesis under test
The LLM-discovered schedules generalize to a third, geometrically
distinct polygon never seen during training and not in the tuning loop.

## Method
Construct one new held-out cell. Two natural candidates:

1. **Horns Rev I geometry** (real, well-known): rectangular grid layout
   with N=80 turbines, well-published. Use only the polygon and turbine
   count; pick a wind rose distinct from DEI and ROWP.
2. **Synthetic non-convex polygon**: an L-shape or a polygon with a
   concave bay. Stress-tests boundary penalty under non-convex SDF.

We run both, but report option 2 in the paper (it is the cleaner stress
test and avoids real-data licensing/comparison messiness).

For each new cell:
- Generate `results/problem_<name>.json`.
- Run a 500-multi-start TopFarm SGD baseline (same protocol as DEI/ROWP).
- Score top Claude/Gemini schedules.
- Score top SLSQP and bump-family DE for context.

## Cost
- Polygon generation: minutes.
- Baseline 500-start: ~6 hr CPU (similar to ROWP baseline).
- Schedule eval: 4 schedules × 30–340s ≈ 30 min.
- Optionally: full 64-cell extension for new polygon (skip for week-2).

## Inputs
- `tools/build_synthetic_polygon.py` (to be written)
- `tools/build_baseline.py` (existing pattern; copy-adapt)

## Outputs
- `results/problem_lshape.json`
- `results/baseline_lshape.json` (mean, best, layouts)
- `experiments/F_extra_heldout/results.json` — top schedules' AEP

## Success criteria
- Schedule-only LLM scores beat baseline on the new cell. If they
  don't, generalization claim weakens; report honestly.
- The relative ranking (Claude vs Gemini, schedule-only vs full-opt)
  is consistent with DEI/ROWP. Inconsistent ranking suggests the
  observed "Claude > Gemini" or "schedule-only ≥ full-opt" was a
  polygon-specific artifact.

## Launch
```
bash experiments/F_extra_heldout/launch.sh
```

## Aggregation
```
pixi run python experiments/F_extra_heldout/aggregate.py
```
