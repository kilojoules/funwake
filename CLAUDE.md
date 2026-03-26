# FunWake — LLM-Generated Layout Optimizers

FunSearch-style loop: an LLM writes wind farm layout optimization code,
a benchmark suite scores it, the score feeds back, the LLM refines.

## The experiment

**Task**: Write a constrained optimizer that maximizes AEP for turbine
layouts inside polygon boundaries. All farms share the same turbine
(IEA 15 MW, D=240m) and 10-year DEI wind resource.

**Train/test split**:
- Training: DEI farm 1 (50 turbines, IEA 15 MW, D=240m). The LLM
  develops its optimizer on this farm, sees boundary and results.
- Test (held out): IEA Wind 740-10 ROWP irregular layout case
  (74 turbines, IEA 10 MW, D=198m, different polygon + Weibull wind).
  The LLM NEVER sees this. Problem: `results/problem_rowp.json`.
- Validation: DEI farm 0 (same shape as farm 1). Quick sanity check.
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
pixi run python agent.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --n-iters 10
```

## Architecture

```
agent.py           — multi-turn LLM loop (Together AI)
  ↓ writes
playground/        — LLM's pixwake clone (can read/modify)
  ↓ runs script on training farms
benchmarks/        — firewalled scorer (LLM can't touch)
  ↓
AEP per farm       — fed back to LLM; test farm scored at end
```

## Benchmark cases

| Case | Turbines | Turbine | Split |
|------|----------|---------|-------|
| DEI farm 1 | 50 | IEA 15MW (D=240m) | train |
| DEI farm 0 | 50 | IEA 15MW (D=240m) | validation |
| ROWP irregular | 74 | IEA 10MW (D=198m) | TEST (held out) |

The ROWP case (from IEA Wind 740-10-ROWP) has a different polygon,
turbine, and Weibull wind resource — tests true generalization.

## Key files

- `benchmarks/dei_layout.py` — 10-farm benchmark suite + baseline
- `agent.py` — FunSearch multi-turn loop
- `playground/pixwake/` — LLM's pixwake clone (created by setup.sh)
- `results/` — baselines, generated scripts, history

## Cost

Llama 3.3 70B Turbo via Together AI: ~$0.88/M tokens.
10 iterations ≈ $0.10-0.50.
