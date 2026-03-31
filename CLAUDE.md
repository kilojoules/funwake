# FunWake — LLM-Generated Layout Optimizers

An LLM agent writes wind farm layout optimization code, competing
against a 500 multi-start baseline.

## The experiment

**Task**: Write a general constrained optimizer that maximizes AEP for
turbine layouts inside polygon boundaries. The script reads all
parameters (turbine, boundary, wind) from a problem JSON and must
generalize to unseen farms.

**Train/test split**:
- Training: DEI farm 1 (50 turbines, IEA 15 MW, D=240m).
- Test (held out): IEA Wind 740-10 ROWP irregular layout case
  (74 turbines, IEA 10 MW, D=198m, different polygon + Weibull wind).
  The LLM never sees ROWP's AEP. Problem: `results/problem_rowp.json`.

**Baseline**: 500 multi-start topfarm_sgd_solve (max_iter=4000,
additional_constant_lr_iterations=2000).

## Setup

```bash
pixi install
bash setup.sh  # clones pixwake, runs baselines
```

## Run

```bash
pixi run python agent_cli.py \
    --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
    --provider gemini --model gemini-2.5-flash \
    --time-budget 3600 \
    --hot-start results/seed_optimizer.py
```

## Architecture

```
agent_cli.py       — agentic tool-use loop (Gemini function calling)
  ↕ tools (read_file, write_file, run_tests, run_optimizer, test_generalization)
playground/        — LLM workspace (pixwake source, problem.json, test suite)
benchmarks/        — firewalled scorer (LLM can't touch)
results/           — baselines, problem definitions
```

## Key files

- `agent_cli.py` — main entry point
- `playground/test_optimizer.py` — unit test suite
- `playground/problem.json` — training farm definition
- `results/seed_optimizer.py` — generic baseline template
- `results_agent_1hr_v7/best_optimizer.py` — best LLM-generated optimizer
- `plot_progress.py` — progress visualization
