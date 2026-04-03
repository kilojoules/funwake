# FunWake: LLM-Generated Wind Farm Layout Optimizers

An LLM agent (Gemini 2.5 Flash) autonomously writes wind farm layout
optimization code, competing against a strong 500-multi-start gradient
baseline. Given tools to explore a wake simulation codebase, write
optimizer functions, run tests, and receive scores, the agent discovers
**wind-direction-aware grid initialization** — a domain insight that
aligns turbine rows perpendicular to the prevailing wind, reducing wake
losses before optimization begins.

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
optimizer the LLM produced, generated autonomously by Gemini 2.5 Flash.

The winning strategy combines three elements developed over 5 hours
of autonomous iteration:

1. **Wind-direction-aware grid initialization.** Computes the
   energy-weighted dominant wind direction, rotates the turbine
   placement grid perpendicular to it. Turbines start in positions
   that naturally minimize wake interference.

2. **Diverse multi-start pool.** Three initialization strategies —
   wind-aware grid, standard grid, random — give the optimizer
   different basins of attraction to explore.

3. **High constraint penalties with tuned SGD.** `spacing_weight=75,
   boundary_weight=75, ks_rho=80, learning_rate=150` — aggressive
   penalties ensure feasibility while the high learning rate enables
   large layout rearrangements.

## Progress Over Time

![Agent progress](results_agent_5hr_v4/progress.png)

99 attempts over 5 hours: 59 successful, 17 custom optimizer attempts
(after a phase-2 exploration nudge), 40 errors. Training AEP climbs
above the baseline; held-out ROWP tracks alongside.

## Key Findings

### 1. The LLM tunes hyperparameters, not algorithms

Across all runs, the LLM's winning strategies are variations of the
provided `topfarm_sgd_solve` optimizer with different settings (learning
rate, penalty weights, iteration counts) and initialization strategies.
When nudged toward custom optimizers (phase-2 prompting with an Adam
template), it produces code that scores higher on training but fails
constraint checks on the held-out farm.

| Strategy | Training AEP | ROWP (held-out) | Feasible? |
|----------|-------------|-----------------|-----------|
| sgd_solve wrappers (42 attempts) | best: 5563 | best: 4264 | 96% |
| Custom optimizers (17 attempts) | best: 5600 | best: 4194 | ~50% |

Custom optimizers score higher on training but fail to generalize.
The root cause: `topfarm_sgd_solve` uses adaptive penalty ramping
(alpha increases as learning rate decays), while custom optimizers
use fixed or decreasing penalties that allow constraint drift on
tighter polygons.

### 2. The LLM discovers strategies, not algorithms

The LLM's genuine contributions are at the strategy level:
- **Wind-direction-aware initialization** (a real domain insight)
- **Two-stage optimization** (feasibility then AEP)
- **Diverse multi-start** with perturbation scaling relative to
  `min_spacing` (generalizes across farm sizes)

These are optimization *strategies* that compose existing building
blocks, not novel optimization *algorithms*.

### 3. Constraint handling is the generalization bottleneck

The stressed polygon unit test (thin rhombus, tight packing) catches
optimizers that work on spacious training farms but produce NaN or
constraint violations on harder geometries. Every custom optimizer
that failed on ROWP also failed this test — it's an effective filter.

## Future Directions

- **Remove `topfarm_sgd_solve`**: Force the LLM to write optimization
  from scratch. Would it discover proper penalty ramping independently?
- **Multiple training farms**: Currently 1 training farm. Adding 2-3
  with different geometries would improve generalization pressure.
- **Ablation**: Compare LLM vs systematic grid search over SGDSettings
  hyperparameters with the same time budget.
- **Multiple independent runs**: 3-5 runs for confidence intervals.
- **Multiple LLMs**: Compare Gemini Flash vs Claude vs GPT-4o.

## How It Works

The LLM writes ONLY an `optimize()` function:

```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Returns (opt_x, opt_y) — optimized turbine positions."""
```

A harness handles physics (wake model fixed at k=0.04, turbine from
JSON). The LLM cannot modify the wake model or game the scorer.

### Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read pixwake source code, training problem JSON |
| `write_file` | Save optimizer scripts to workspace |
| `run_tests` | Signature check, 3-turbine quick test, stressed polygon test |
| `run_optimizer` | Score on training farm — returns AEP |
| `test_generalization` | Held-out farm PASS/FAIL (no AEP leaked) |
| `get_status` | Best AEP vs baseline |

### Sandbox

Generated code runs in `sandbox-exec`: no network, stripped environment
(no API keys), filesystem writes restricted to workspace. Static
blocklist for subprocess, exec, eval.

### Search guidance

- **Phase-2 prompting**: after 30% of time, provides an Adam template
  and nudges toward custom optimizers
- **Diversity nudge**: after 5 consecutive `topfarm_sgd_solve`
  submissions, suggests alternatives
- **Context pruning**: compresses conversation after 40 turns
- **60s explore timeout**: prevents multi-start bloat during iteration

### Unit tests

The LLM can run `run_tests --quick` for fast validation (~10s):

| Test | What it catches |
|------|----------------|
| Signature check | Wrong function parameters |
| Quick run (3 turbines) | Crashes, wrong count, NaN |
| **Stressed polygon** (25 turbines, thin rhombus) | **Weak constraints on tight geometry** |

The stressed polygon test is the key filter — it catches every custom
optimizer that later fails on the held-out farm.

## Background

Wind turbines create wakes — regions of slower air behind each rotor.
Layout optimization places N turbines inside a polygon to maximize
annual energy production (AEP), subject to minimum spacing constraints.
The problem is non-convex with many local optima.

### Benchmark cases

| Case | Turbines | Turbine | Baseline | Role |
|------|----------|---------|----------|------|
| DEI farm 1 | 50 | IEA 15MW, D=240m | 5540.72 GWh | Training |
| [IEA ROWP](https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP) | 74 | IEA 10MW, D=198m | 4246.67 GWh | Held-out test |

Baselines: 500 multi-start `topfarm_sgd_solve` with grid initialization.

## Methodology Notes

Building a fair benchmark required solving several subtle problems:

- **Non-convex polygon**: The ROWP boundary was non-convex, causing
  all evaluations to silently fail. Fixed by convex hull before saving.
- **Wake model gaming**: The LLM changed k=0.04 to k=0.0505 to inflate
  scores. Fixed by moving physics into the harness.
- **Double-indexing bug**: `grid_y[perm[perm[:n]]]` worked on DEI but
  produced overlapping turbines on ROWP. Caught by the test suite.
- **Identical polygons**: All 10 DEI farms were translated copies.
  Fixed by introducing the genuinely different ROWP farm.
- **Unfair baseline**: ROWP baseline used a pre-optimized reference
  layout. Recomputed with grid initialization.

## Reproduce

### Prerequisites

- [pixi](https://pixi.sh), Gemini API key in `~/.gem`

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

## Repository Structure

```
agent_cli.py                  Agentic tool-use loop (main entry point)
setup.sh                      Clone pixwake + compute baselines
plot_progress.py              Progress visualization

playground/
  harness.py                  Calls optimize() with fixed physics
  test_optimizer.py           Unit tests (signature, quick, stressed polygon)
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
  best_optimizer.py           Best from 5-hour run
  iter_050.py                 Best feasible held-out result
  attempt_log.json            99-attempt history with paired scores
  progress.png                Training vs held-out AEP over time
```
