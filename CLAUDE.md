# FunWake — LLM-Generated Layout Optimizers

FunSearch-style loop: an LLM writes wind farm layout optimization code,
a benchmark suite scores it, the score feeds back, the LLM refines.

## The experiment

**Task**: Write a constrained optimizer that maximizes AEP for turbine
layouts inside polygon boundaries. The optimizer must read all
parameters (turbine, boundary, wind) from a problem JSON so it
generalizes across different farms.

**Train/test split**:
- Training: DEI farm 1 (50 turbines, IEA 15 MW, D=240m). The LLM
  develops its optimizer on this farm, sees boundary and results.
- Test (held out): IEA Wind 740-10 ROWP irregular layout case
  (74 turbines, IEA 10 MW, D=198m, different polygon + Weibull wind).
  The LLM NEVER sees this. Problem: `results/problem_rowp.json`.
- Note: All 10 DEI polygons are the same centered shape, so only
  farm 1 is used for training.

**Baseline**: topfarm_sgd_solve — 500 multi-start constrained ADAM
with LR decay, KS-aggregated penalties, max_iter=4000,
additional_constant_lr_iterations=2000. The LLM tries to beat it.

**What the LLM can do**: Full access to its own pixwake clone. It can
use topfarm_sgd_solve, write its own optimizer, try multistart, modify
constraint handling, use scipy.optimize, etc.

**Scoring**: The benchmark evaluates the resulting layout via pixwake
wake sim. Fast (~1-2s per farm per score).

## Setup

```bash
pixi install
bash setup.sh  # clones pixwake, runs baselines on all farms
```

## Run

```bash
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 600
```

## Architecture

```
agent_cli.py       — agentic tool-use loop (Gemini function calling)
  ↕ tools
playground/        — LLM's pixwake clone (read-only via tools)
  ↓ runs sandboxed script on training farm
benchmarks/        — firewalled scorer (LLM can't touch)
  ↓
AEP score          — fed back to LLM; ROWP scored at end
```

## Benchmark cases

| Case | Turbines | Turbine | Split |
|------|----------|---------|-------|
| DEI farm 1 | 50 | IEA 15MW (D=240m) | train |
| ROWP irregular | 74 | IEA 10MW (D=198m) | TEST (held out) |

The ROWP case (from IEA Wind 740-10-ROWP) has a different polygon,
turbine, and Weibull wind resource — tests true generalization.

## Key files

- `agent_cli.py` — agentic tool-use loop (main entry point)
- `agent.py` — fixed-loop agent (earlier iteration)
- `benchmarks/dei_layout.py` — benchmark suite + baseline runner
- `benchmarks/build_rowp_problem.py` — builds ROWP test case
- `playground/pixwake/` — LLM's pixwake clone (created by setup.sh)
- `results/` — baselines, problem definitions, generated scripts

## Cost

Gemini 2.5 Flash: ~$0.05 per 10-minute session.
Compute: ~20s per optimizer run on MacBook CPU.
