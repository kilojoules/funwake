# Paper Outline: Can an LLM Find a Better Wind Farm Optimization Algorithm?

## Target venues
- ICML 2026 AI Scientists Workshop (4-8 pages, deadline Apr 21-24)
- NeurIPS 2026 Evaluations & Datasets Track (full paper, deadline May 4-6)

---

## Title options
1. "LLM Agents as Optimization Algorithm Designers: An Evaluation Framework for Wind Farm Layout"
2. "Strategies Without Algorithms: What LLM Agents Actually Discover for Engineering Optimization"
3. "Do LLM Agents Design Algorithms or Tune Hyperparameters? Evidence from Wind Farm Layout Optimization"

## Abstract (~150 words)

We present an evaluation framework for testing whether LLM agents can
autonomously design optimization algorithms for real engineering problems.
An LLM receives a wake simulation API, a training farm, and a time
budget. It writes optimizer functions evaluated on the training farm,
with silent scoring on a held-out farm with a different turbine, polygon,
and wind resource. Across three frontier models (Gemini 2.5 Flash,
Claude, Llama 3.1 405B), we find that LLMs consistently discover
effective optimization *strategies* — wind-direction-aware initialization,
two-stage feasibility-then-objective optimization, diverse multi-start —
but never produce novel optimization *algorithms*. All winning solutions
wrap the existing gradient solver with tuned hyperparameters. Custom
gradient implementations fail constraint checks on the held-out farm.
A stressed-polygon unit test predicts held-out failure with high accuracy.

---

## 1. Introduction

- The promise: LLMs as autonomous algorithm designers (FunSearch, AlphaEvolve)
- The gap: most results are on synthetic/mathematical problems, not constrained engineering
- Our question: can an LLM design a layout optimizer that generalizes to unseen farms?
- Preview of findings: strategies yes, algorithms no

## 2. Related Work

- FunSearch (Nature 2023): program search for mathematical discovery
- AlphaEvolve (2025): Gemini-powered algorithm evolution
- ReEvo (NeurIPS 2024): LLMs as hyper-heuristics
- HeuriGym, CO-Bench: benchmarks for LLM optimization agents
- Wind farm layout optimization: gradient methods, evolutionary, hybrid
- Gap: no rigorous held-out evaluation framework for LLM-generated engineering code

## 3. Evaluation Framework

### 3.1 Problem formulation
- Maximize AEP subject to boundary + spacing constraints
- Function interface: optimize(sim, n_target, boundary, min_spacing, wd, ws, weights)
- Harness handles physics (fixed wake model) — LLM cannot game the scorer

### 3.2 Train/test split
- Training: DEI farm (50 turbines, IEA 15MW, D=240m, 24-sector wind rose)
- Held-out: IEA ROWP (74 turbines, IEA 10MW, D=198m, 12-sector Weibull wind)
- LLM sees training AEP; held-out only PASS/FAIL feasibility (no AEP leaked)

### 3.3 Baselines
- 500 multi-start topfarm_sgd_solve (max_iter=4000, const_lr=2000)
- Fair baselines: grid initialization, no pre-optimized layouts

### 3.4 Unit test suite
- Signature check (<1s)
- Quick 3-turbine test (<3s)
- Stressed polygon: thin rhombus, 25 turbines, tight packing (~7s)
- Full farm test via harness (~25s)

### 3.5 Agent architecture
- Tool-use loop with read/write/test/score/generalize tools
- Phase-2 prompting: after 30% of time, nudge toward custom optimizers
- Diversity nudge after 5 consecutive same-strategy submissions
- Context pruning for long runs
- Sandboxed execution (no network, stripped env)

## 4. Experiments

### 4.1 Models tested
- Gemini 2.5 Flash (API)
- Claude via Claude Code (CLI)
- Llama 3.1 405B (self-hosted vLLM on LUMI)

### 4.2 Experimental setup
- 3-5 seeds per model
- 5-hour time budget per run
- 60s explore timeout, 300s final eval timeout
- Hot-start from seed optimizer template

### 4.3 Metrics
- Training AEP (GWh) vs baseline
- Held-out AEP (GWh) vs baseline
- Feasibility rate on held-out farm
- Strategy classification (sgd_solve wrapper vs custom)
- Time to first improvement

## 5. Results

### 5.1 Multi-model progress curves
- Figure 1: Training + held-out AEP over time, all models, best-of-seeds
  (the key figure — shows convergence behavior and generalization)

### 5.2 Strategy analysis
- Table: breakdown of sgd_solve vs custom attempts per model
- Custom optimizers score higher on training but fail on held-out
- Root cause: penalty annealing direction (high→low vs low→high)

### 5.3 What the LLMs discovered
- Wind-direction-aware grid initialization
- Two-stage optimization (feasibility then AEP)
- Diverse multi-start pools
- All are *strategies*, not *algorithms*

### 5.4 What the LLMs failed to discover
- Proper penalty ramping (alpha increases as lr decays)
- ADAM normalization for constraint gradients
- Any approach that doesn't wrap topfarm_sgd_solve

### 5.5 Ablation: LLM vs systematic hyperparameter search
- Grid search over SGDSettings with same time budget
- Does the LLM's strategy discovery outperform brute-force tuning?

### 5.6 Stressed polygon test as a predictor
- Every custom optimizer that failed on held-out also failed stressed test
- Precision/recall of the test as a filter

## 6. Discussion

- Why strategies but not algorithms? The LLM has a working solver available
  and no incentive to take the risk of writing one from scratch
- The constraint handling gap: LLMs don't discover penalty ramping because
  the training farm is spacious enough to mask weak constraints
- Implications for LLM-as-algorithm-designer: the action space matters —
  giving the LLM an existing solver creates an attractive local optimum
- The evaluation framework as the contribution: held-out generalization
  is essential but absent from prior work

## 7. Conclusion

- LLMs are effective optimization *tuners*, not algorithm *designers*
- The evaluation framework (train/test, stressed polygon, sandboxed
  execution) is necessary to distinguish the two
- The wind-direction-aware initialization is a genuine discovery
- Future: remove existing solver, multiple training farms, evolutionary
  population of LLM agents

---

## Figures

1. **Multi-model progress** (key figure): 2-panel plot, training AEP (top)
   and held-out AEP (bottom) vs time. One line per model (best-of-seeds),
   with shaded CI bands. Baseline reference lines.

2. **Strategy breakdown**: stacked bar chart — sgd_solve vs custom attempts
   per model, colored by feasibility on held-out.

3. **Architecture diagram**: agent loop with tools, harness, scorer.

4. **Stressed polygon test**: the thin rhombus polygon with example
   feasible vs infeasible layouts.

5. **Ablation**: LLM best vs grid search best, training and held-out.

## Tables

1. **Main results**: model × {training AEP, held-out AEP, feasibility rate,
   n_attempts, strategy mix} with 95% CI from seeds.

2. **Best optimizer comparison**: side-by-side code snippets of LLM's best
   vs baseline, highlighting the discovered strategies.
