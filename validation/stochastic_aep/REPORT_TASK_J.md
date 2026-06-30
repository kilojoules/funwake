# Task J — α-weight ablation under K=50 stochastic gradient

Replaces `tools/alpha_ablation.py` (coarse deterministic gradient via
`tools/run_optimizer.py`) with K=50 categorical-rose Monte Carlo
inner gradient under iter_192 schedule with α scaled by FACTOR.

## Setup

- Schedule: `funwake_iter192_alpha_scaled(factor)` in
  `validation/stochastic_aep/schedules_ablation.py` —
  iter_192 LR + bumps + β=(0.3, 0.5) unchanged; α multiplied by
  FACTOR.
- Factors: 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0 (9 levels).
- Problems:
  - DEI training (`playground/problem.json`, N=50, Bastankhah k=0.04)
  - ROWP held-out (`results/problem_rowp.json`, N=74, Bastankhah k=0.04
    — matches `playground/harness.py` deployment regime, NOT the
    physically-correct NOJ k=0.05 used in the 740-10 validation work).
- K=50 categorical-rose AEP gradient, ES-off, 8000 SGD steps.
- 3 sample seeds per (factor, problem): 100000, 200000, 300000.
- Total runs: 54. Wall ≈ 50 min (DEI 2-7s/run, ROWP 2-3s/run).

## Bastankhah results (paper-comparable regime)

| factor | DEI mean | ROWP feas-mean | feas | ROWP best |
|-------:|---------:|---------------:|-----:|----------:|
|  0.01  | 5563.65  | 4267.92        | 3/3  | 4269.21   |
|  0.05  | 5564.87  | 4264.78        | 3/3  | 4265.43   |
|  0.10  | 5564.80  | 4267.64        | 3/3  | 4268.52   |
|  0.25  | 5562.63  | 4262.89        | 3/3  | 4264.90   |
|  0.50  | 5564.05  | 4264.46        | 3/3  | 4265.84   |
|  1.00  | 5560.47  | 4263.35        | 3/3  | 4264.87   |
|  2.00  | 5560.14  | 4264.57        | 3/3  | 4265.31   |
|  5.00  | 5558.97  | 4264.13        | 3/3  | 4265.14   |
| 10.00  | 5558.98  | 4265.22        | 3/3  | 4268.09   |

- All 27 ROWP runs feasible (3/3 per factor) — confirms paper claim
  "all nine variants produce feasible held-out layouts".
- Feasible-mean span: 4262.89 (factor=0.25) to 4267.92 (factor=0.01)
  = **5.0 GWh** (vs paper's old 11.2 GWh under coarse gradient).
- Peak shifts from factor=1.0 (old) to factor=0.01 (new K=50).
- Best at factor=1.0: 4264.87 GWh (deployed iter_192 was 4271.5 via
  full-init multi-start harness).

## NOJ results (physically-correct 740-10 regime; archived)

- `task_j_alpha_k50_noj.json`. ROWP AEPs 4074-4138.
- Feasibility NOT uniform: 0-3/3 across factors (low α has trouble
  reaching feasibility under stochastic gradient with NOJ wake).
- NOT comparable to paper because harness.py uses Bastankhah; archived
  for reference only.

## Paper impact

Updated 4 paragraphs:
- `paper/short.tex` L355 — "penalty weight affects quality, not feasibility"
- `paper/main.tex` L121 — same paragraph in body
- `paper/sections/discussion.tex` L39 — "Penalty weight affects quality only weakly"
- `paper/sections/introduction.tex` L71 — penalty trajectory mention

Qualitative conclusions preserved:
- ✓ All 9 factors feasible — confirmed under K=50
- ✓ LR dynamics drive feasibility — confirmed
- △ Quality range: 11.2 GWh → 5.0 GWh (narrower, peak shifts)

## Files

- `run_j_alphaweight.py` — main runner
- `run_j_sanity_es.py` — ES-on vs ES-off sanity (factor=1.0, ROWP, NOJ)
- `schedules_ablation.py` — `funwake_iter192_alpha_scaled` added
- `task_j_alpha_k50.json` + `.log` — Bastankhah results (paper-comparable)
- `task_j_alpha_k50_noj.json` + `.log` — NOJ results (archived)
