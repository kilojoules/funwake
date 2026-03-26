# FunWake: Can an LLM Write a Better Wind Farm Optimizer?

An experiment where an LLM agent autonomously writes wind farm layout
optimization code, competing against a strong multi-start baseline.
The agent has tools to explore the codebase, write optimizer scripts,
run them, and iterate — similar to a developer using an AI coding
assistant.

## Results

The LLM-generated optimizer beats a 500 multi-start baseline by
**+40 GWh** on the training farm and **+38 GWh** on the held-out
validation farm (same polygon shape, different initial conditions).

| Case | Baseline (GWh) | LLM Best (GWh) | Gap |
|------|---------------|-----------------|-----|
| DEI farm 1 (train) | 5540.72 | **5580.95** | **+40.23** |
| DEI farm 0 (validation) | 5542.14 | **5580.52** | **+38.38** |

### What the LLM discovered

The winning optimizer (`results_agent_2hr/best_optimizer.py`) uses three
ideas that the baseline lacks:

1. **Two-stage optimization** — First stage uses heavy constraint
   penalties (spacing_weight=50, boundary_weight=50, ks_rho=750) to find
   a feasible layout. Second stage optimizes AEP with relaxed penalties.
   The baseline uses a single stage with default penalty weights.

2. **Tuned Adam momentum** — beta1=0.95, beta2=0.99 (close to standard
   Adam) outperforms TopFarm's defaults of beta1=0.1, beta2=0.2. This
   was the biggest single improvement, discovered at turn 23 of the
   2-hour session.

3. **Hybrid multi-start** — 20 starts split between random grid-based
   layouts (exploration) and perturbations of the current best
   (exploitation). The baseline uses 500 pure random starts but still
   loses because of (1) and (2).

## Problem

Optimize the placement of ~50 wind turbines (IEA 15 MW, D=240m) inside
a convex polygon boundary to maximize Annual Energy Production (AEP).

**Constraints:**
- All turbines inside the polygon (`boundary_penalty < 1e-3`)
- Minimum spacing 4×D = 960m between any pair

**Wind resource:** 10-year daily-averaged timeseries from the Danish
Energy Island cluster, binned into 24 directional sectors.

**Wake model:** Bastankhah Gaussian deficit (k=0.04) via
[pixwake](https://github.com/kilojoules/cluster-tradeoffs) (JAX).

## Benchmark cases

The 10 DEI polygons in `benchmarks/dei_layout.py` are all the same shape
(translated copies in UTM space that become identical after centering).
Only two are used:

| Case | Role | Notes |
|------|------|-------|
| DEI farm 1 | **Training** | LLM develops and tests here |
| DEI farm 0 | **Validation** | Same polygon — LLM never sees it during prototyping |

An additional held-out test case (IEA Wind 740-10 ROWP, 74 turbines,
IEA 10MW, different polygon and Weibull wind resource) is defined in
`results/problem_rowp.json` with a baseline in `results/baseline_rowp.json`,
but has not yet been used for evaluation. It is intended for future
generalization testing with a different turbine and site.

## Baseline

500 multi-start `topfarm_sgd_solve` with `max_iter=4000`,
`additional_constant_lr_iterations=2000`, `learning_rate=50.0`.
Best of 500 random initial layouts. Takes ~2.5 hours per farm.

## Agent

The agent (`agent_cli.py`) is a tool-use loop powered by Gemini 2.5
Flash via function calling. It has four tools:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read pixwake source code (restricted to playground/) |
| `list_files(path)` | Explore the codebase |
| `run_optimizer(code)` | Run a complete optimizer script on the training farm, get AEP back |
| `get_status()` | Check best AEP and gap vs baseline |

### Sandbox

Generated optimizer scripts run inside a security sandbox:

- **macOS sandbox-exec** blocks all network access and restricts
  filesystem writes
- **Stripped environment** — only whitelisted env vars (no API keys)
- **Static safety checker** — blocklist for subprocess, exec, network,
  filesystem manipulation
- **Restricted file access** — blocks .git, .env, binary files

### Hot start

The `--hot-start` flag seeds the agent with a known-good optimizer
script so it skips API discovery and starts optimizing immediately.
The 2-hour winning run used the best script from an earlier 5-iteration
Gemini session as its seed.

## Reproduce

### Prerequisites

- [pixi](https://pixi.sh) package manager
- Gemini API key in `~/.gem`
- Wind resource CSV (contact authors or use your own)

### Setup

```bash
pixi install
bash setup.sh  # clones pixwake, runs 500-start baselines (~5 hours)
```

### Run the agent

```bash
# 10-minute session with hot start
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 600 \
    --hot-start results_gemini/best_optimizer.py

# 2-hour session (what produced the +40 GWh result)
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 7200 \
    --hot-start results_gemini/best_optimizer.py \
    --output-dir results_2hr
```

## Key files

| File | Description |
|------|-------------|
| `agent_cli.py` | Agentic tool-use loop (main entry point) |
| `agent.py` | Fixed-loop agent (earlier iteration) |
| `benchmarks/dei_layout.py` | Benchmark suite, baseline runner, scorer |
| `benchmarks/build_rowp_problem.py` | Builds ROWP test case from IEA data |
| `setup.sh` | Clones pixwake, computes baselines |
| `results/baselines.json` | 500-start baseline (AEP + layouts, farms 0 & 1) |
| `results/problem_farm{0,1}.json` | DEI problem definitions |
| `results/problem_rowp.json` | ROWP test case (not yet evaluated) |
| `results_gemini/best_optimizer.py` | Best from 5-iteration run (+37 GWh) |
| `results_agent_2hr/best_optimizer.py` | Best from 2-hour session (+40 GWh) |

## Cost

- Gemini 2.5 Flash: ~$0.05 per 10-minute session
- Compute: ~20s per optimizer run on MacBook CPU (M-series)
- Baseline precompute: ~2.5 hours per farm (500 multi-starts)
