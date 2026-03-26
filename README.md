# FunWake: Can an LLM Write a Better Wind Farm Optimizer?

An experiment where an LLM agent autonomously writes wind farm layout
optimization code, competing against a strong multi-start baseline
optimizer. The agent has tools to explore the codebase, write scripts,
run them, and iterate — similar to how a developer would use an AI
coding assistant.

## Results

The best LLM-generated optimizer beats a 500 multi-start baseline by
**+40 GWh** on the training farm and **+38 GWh** on the held-out
validation farm.

| Case | Baseline (GWh) | LLM Best (GWh) | Gap |
|------|---------------|-----------------|-----|
| DEI farm 1 (train) | 5540.72 | **5580.95** | **+40.23** |
| DEI farm 0 (validation) | 5542.14 | **5580.52** | **+38.38** |

### What the LLM discovered

The winning optimizer (`results_agent_2hr/best_optimizer.py`) uses three
innovations over the baseline:

1. **Two-stage optimization** — First stage uses heavy constraint
   penalties (spacing_weight=50, boundary_weight=50, ks_rho=750) to find
   a feasible layout. Second stage optimizes AEP with relaxed penalties.
   The baseline uses a single stage with default penalty weights.

2. **Tuned Adam momentum** — The agent found that beta1=0.95, beta2=0.99
   (close to standard Adam defaults) outperforms the TopFarm defaults of
   beta1=0.1, beta2=0.2. This was the biggest single improvement
   (+1 GWh), discovered at turn 23 of the 2-hour session.

3. **Hybrid multi-start** — 20 starts split between random grid layouts
   (exploration) and perturbations of the current best (exploitation).
   The baseline uses 500 pure random starts.

### What this means

The LLM didn't invent a new optimization algorithm. It discovered that
(a) separating feasibility from objective optimization, (b) using
standard Adam momentum instead of TopFarm's custom values, and (c)
smart initialization beats brute-force random restarts. These are
reasonable engineering insights that a human optimizer would eventually
find, but the LLM found them autonomously in 2 hours.

## Experiment setup

### Problem

Optimize the placement of ~50 wind turbines (IEA 15 MW, D=240m) inside
a polygon boundary to maximize Annual Energy Production (AEP). Constraints:
all turbines inside the boundary, minimum spacing 4×D = 960m.

### Baseline

500 multi-start `topfarm_sgd_solve` (constrained Adam with KS-aggregated
penalties): `max_iter=4000, additional_constant_lr_iterations=2000,
learning_rate=50.0`. Best of 500 random initial layouts. Takes ~2.5 hours
per farm.

### Benchmark cases

| Case | Turbines | Turbine | Role |
|------|----------|---------|------|
| DEI farm 1 | 50 | IEA 15MW (D=240m) | Training |
| DEI farm 0 | 50 | IEA 15MW (D=240m) | Validation |
| ROWP irregular | 74 | IEA 10MW (D=198m) | Held-out test (different turbine + polygon) |

The DEI farms use a 10-year daily-averaged wind resource from the Danish
Energy Island cluster. The ROWP case uses a Weibull wind resource from
the [IEA Wind 740-10 ROWP](https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP).

### Agent

The agent (`agent_cli.py`) is a tool-use loop powered by Gemini 2.5
Flash. It has four tools:

- `read_file` — inspect pixwake source code
- `list_files` — explore the codebase
- `run_optimizer` — write and run a complete optimizer script, get AEP back
- `get_status` — check best AEP vs baseline

The agent runs inside a sandbox:
- **macOS sandbox-exec** blocks all network access and restricts filesystem
  writes to the playground directory
- **Stripped environment** — only whitelisted env vars (no API keys, no
  cloud credentials)
- **Static safety checker** — blocklist for subprocess, exec, network,
  filesystem manipulation (no LLM-based safety check that could hallucinate)
- **Read-only file access** — blocks .git, .env, binary files, credentials

### Time budget

- **Prototyping**: 10 minutes (or longer via `--time-budget`)
- **Per-farm timeout**: 5 minutes (300s)
- **Hot start**: optionally seed the agent with a known-good script so it
  skips API discovery and jumps straight to optimization

## Reproduce

### Prerequisites

- [pixi](https://pixi.sh) package manager
- Gemini API key in `~/.gem`
- Wind resource CSV (10-year Danish Energy Island data)

### Setup

```bash
pixi install
bash setup.sh  # clones pixwake, runs 500-start baselines (~5 hours)
```

### Run the agent

```bash
# 10-minute session with hot start from previous best
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 600 \
    --hot-start results_gemini/best_optimizer.py \
    --output-dir results_my_run \
    --run-id myrun

# 2-hour session (what produced the best result)
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 7200 \
    --hot-start results_gemini/best_optimizer.py \
    --output-dir results_2hr
```

### Run the old-style fixed-loop agent

```bash
pixi run python agent.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --n-iters 5 --output-dir results_loop
```

### Score a layout

```bash
PYTHONPATH=playground/pixwake/src pixi run python benchmarks/dei_layout.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    score --farm-id 1 --layout results_agent_2hr/best_layout.json
```

## Architecture

```
agent_cli.py          Agentic tool-use loop (Gemini function calling)
  ↕ tools
playground/pixwake/   LLM's pixwake clone (read-only via tools)
  ↓ runs sandboxed script
benchmarks/           Firewalled scorer (LLM can't touch)
  ↓
AEP score             Fed back to LLM
```

### Key files

| File | Description |
|------|-------------|
| `agent_cli.py` | Agentic optimizer (Claude Code-style tool-use loop) |
| `agent.py` | Fixed-loop agent (generate → run → score → repeat) |
| `benchmarks/dei_layout.py` | Benchmark suite + baseline runner |
| `benchmarks/build_rowp_problem.py` | Builds ROWP test case from IEA data |
| `setup.sh` | Clones pixwake, runs baselines |
| `results/baselines.json` | 500-start baseline results (AEP + layouts) |
| `results/problem_farm*.json` | Problem definitions for each farm |
| `results/problem_rowp.json` | ROWP held-out test case |
| `results/baseline_rowp.json` | ROWP baseline result |
| `results_gemini/best_optimizer.py` | Best from 5-iteration Gemini run (+37 GWh) |
| `results_agent_2hr/best_optimizer.py` | Best from 2-hour agent session (+40 GWh) |

## Evolution of the approach

1. **v1 (agent.py)**: Fixed generate-run-score loop. LLM-based safety
   checker wasted iterations on false positives. Smaller models (Qwen 7B,
   Llama 70B) couldn't get the API right. Error feedback was too terse.

2. **v2 (agent.py improved)**: Deterministic safety checker, error
   diagnosis with explicit corrections, fail-fast on farm timeouts,
   working template always in prompt. Gemini 2.5 Flash scored +50 GWh
   on the weak (single-start) baseline.

3. **v3 (agent_cli.py)**: Agentic tool-use loop. LLM explores codebase,
   writes code, runs it, iterates freely within a time budget. With hot
   start and 2 hours, Gemini found +40 GWh over the strong 500-start
   baseline.

## Cost

- Gemini 2.5 Flash: ~$0.01-0.05 per 10-minute session
- Compute: ~20s per optimizer run on MacBook CPU (M-series)
- Baseline: ~2.5 hours per farm (500 multi-starts)
