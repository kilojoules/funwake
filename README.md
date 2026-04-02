# FunWake: LLM-Generated Wind Farm Layout Optimizers

An LLM agent (Gemini 2.5 Flash) autonomously writes wind farm layout
optimization code, competing against a strong 500-multi-start gradient
baseline. Given tools to explore a wake simulation codebase, write
optimizer functions, run tests, and receive scores, the agent
independently discovers **wind-direction-aware grid initialization** —
a domain insight that aligns turbine rows perpendicular to the
prevailing wind, reducing wake losses before optimization begins.

## Key Result

|  | Training (DEI, 50 turbines) | Held-out (ROWP, 74 turbines) |
|---|---|---|
| **500-start baseline** | 5540.72 GWh | 4246.67 GWh |
| **LLM best (feasible)** | **5563.49 GWh** (+22.8) | **4264.03 GWh** (+17.4) |

The held-out farm uses a different turbine (IEA 10 MW vs 15 MW), a
different polygon, different turbine count (74 vs 50), and a different
wind resource (Weibull vs observed timeseries). The LLM never sees the
held-out farm's AEP — only PASS/FAIL feasibility. The improvement
generalizes.

The baseline is strong: 500 independent optimization runs, each with
4000 gradient iterations plus 2000 constant-learning-rate iterations.

## The LLM's Best Optimizer

**[`results/best_optimizer.py`](results/best_optimizer.py)** — the best
optimizer the LLM produced, ready to run on any farm. This was generated
autonomously by Gemini 2.5 Flash after 50 iterations of writing,
testing, and refining code.

The winning strategy (developed over 5 hours, 99 attempts):

1. **Wind-direction-aware grid initialization.** The agent computes the
   energy-weighted dominant wind direction, then rotates the turbine
   placement grid so rows run perpendicular to it. Turbines start in
   positions that naturally minimize wake interference — the optimizer
   finds better local optima.

2. **Diverse multi-start pool.** Three initialization strategies —
   wind-aware grid, standard grid, random — give the optimizer
   different basins of attraction to explore.

3. **High constraint penalties with tuned SGD.** `spacing_weight=75,
   boundary_weight=75, ks_rho=80, learning_rate=150` — aggressive
   penalties ensure feasibility while the high learning rate enables
   large layout rearrangements.

The wind-aware initialization is the key insight. A wind energy engineer
would recognize it as sound practice — wake losses are directional, so
aligning the grid to the wind rose is the right thing to do. The LLM
arrived at this independently.

## Progress Over Time

![Agent progress](results_agent_5hr_v4/progress.png)

99 attempts over 5 hours: 59 successful, 17 custom optimizer attempts
(after a phase-2 exploration nudge), 40 errors. Training AEP climbs
above the baseline; held-out ROWP tracks alongside.

## How It Works

The LLM writes ONLY an `optimize()` function:

```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Returns (opt_x, opt_y) — optimized turbine positions."""
```

A harness handles physics (wake model fixed at k=0.04, turbine from
JSON). The LLM cannot modify the wake model or game the scorer.

### Tools

| Tool | What it does |
|------|-------------|
| `read_file` | Read pixwake source code, training problem JSON |
| `write_file` | Save optimizer scripts to workspace |
| `run_tests` | Unit tests: signature, 3-turbine quick check, full farm |
| `run_optimizer` | Score on training farm — returns AEP |
| `test_generalization` | Held-out farm PASS/FAIL (no AEP leaked) |
| `get_status` | Best AEP vs baseline |

### Sandbox

Generated code runs inside `sandbox-exec`: no network, stripped
environment (no API keys), filesystem writes restricted to workspace.
Static blocklist for subprocess, exec, eval.

### Search Guidance

- **Phase-2 prompting**: after 30% of time, nudges toward custom
  optimizers with a working Adam template using `jax.grad`
- **Diversity nudge**: after 5 consecutive `topfarm_sgd_solve`
  submissions, suggests alternative approaches
- **Context pruning**: compresses conversation after 40 turns to
  prevent quality degradation in long runs

## Background

Wind turbines create wakes — regions of slower air behind each rotor.
Downstream turbines in wakes produce less power. Layout optimization
places N turbines inside a polygon to maximize total annual energy
production (AEP), subject to minimum spacing constraints.

This is a non-convex problem with many local optima. The standard
approach: gradient-based optimization from hundreds of random starting
points, keeping the best. The question: can an LLM do better by
reasoning about the problem structure?

AEP is measured in GWh. A 17–23 GWh improvement on a ~5500 GWh farm
is a 0.3–0.4% gain — modest in percentage but significant at industrial
scale.

## Methodology

### Train/test split

| Case | Turbines | Turbine | Baseline | Role |
|------|----------|---------|----------|------|
| DEI farm 1 | 50 | IEA 15MW, D=240m | 5540.72 GWh | Training |
| [IEA ROWP](https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP) | 74 | IEA 10MW, D=198m | 4246.67 GWh | Held-out test |

Baselines: 500 multi-start `topfarm_sgd_solve` with grid initialization
(no pre-optimized reference layouts).

### Debugging stories

Building a fair benchmark required solving subtle problems:

- **Non-convex polygon.** The ROWP boundary was non-convex (6 vertices,
  2 inside the hull). The scorer used a convex hull internally; the
  optimizer used raw vertices. All ROWP evaluations silently produced
  spacing=0. Fixed by hulling before saving.

- **Wake model gaming.** The LLM changed k=0.04 to k=0.0505 to inflate
  AEP scores. Fixed by moving physics into the harness — the LLM
  receives a pre-built simulation it cannot modify.

- **Double-indexing bug.** LLM code: `grid_y[permutation[permutation[:n]]]`.
  Worked on DEI (50 turbines, large polygon) but produced overlapping
  turbines on ROWP (74, tight polygon). Caught by the test suite.

- **Identical polygons.** All 10 DEI farms were translated copies —
  the train/test split was meaningless. Fixed by introducing the
  genuinely different ROWP farm.

- **Unfair baseline.** The ROWP baseline originally used the IEA
  reference layout (pre-optimized). Recomputed with grid initialization.

## Reproduce

### Prerequisites

- [pixi](https://pixi.sh) package manager
- Gemini API key in `~/.gem`
- Wind resource CSV (10-year Danish Energy Island daily averages)

### Setup

```bash
pixi install
bash setup.sh   # Clones pixwake, computes baselines (~5 hours)
```

### Run the agent

```bash
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 3600 \
    --hot-start results/seed_optimizer.py
```

### Plot progress

```bash
pixi run python plot_progress.py results_agent_5hr_v4/attempt_log.json
```

### Run tests on an optimizer

```bash
PYTHONPATH=playground/pixwake/src pixi run python \
    playground/test_optimizer.py my_optimizer.py playground/problem.json
```

## Repository Structure

```
agent_cli.py                  Agentic tool-use loop (main entry point)
setup.sh                      Clone pixwake + compute baselines
plot_progress.py              Progress visualization

playground/
  harness.py                  Calls optimize() with fixed physics
  test_optimizer.py           Unit test suite
  problem.json                Training farm definition

benchmarks/
  dei_layout.py               Baseline runner + scorer
  build_rowp_problem.py       ROWP test case from IEA data

results/
  best_optimizer.py           ★ LLM's best optimizer (wind-aware init)
  seed_optimizer.py           Baseline template (hot-start seed)
  baselines.json              500-start baseline results
  baseline_rowp.json          Held-out baseline
  problem_farm1.json          Training problem definition
  problem_rowp.json           Held-out problem definition

results_agent_5hr_v4/
  best_optimizer.py           LLM's best optimizer (wind-aware init)
  iter_050.py                 Best feasible held-out result
  attempt_log.json            99-attempt history with paired scores
  progress.png                Training vs held-out AEP over time
```
