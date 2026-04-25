# Experiment C — Claude full-optimizer additional seeds

## Reviewer concern
"Of the four (agent × interface) cells, only Gemini × full-optimizer has
n=5 agent-run seeds. Claude × full-optimizer has n=1. The 2×2 narrative
asymmetry is conspicuous."

## Hypothesis under test
Claude's full-optimizer behavior (wrap topfarm_sgd_solve / no SLSQP
rediscovery) is reproducible across agent runs.

## Method
Re-run Claude full-optimizer 4 more times. Same scaffold (Claude Code
CLI via `runners/claude_code_runner.py`), same hot-start, same time
budget (5 hr). The non-determinism comes from API temperature and
timing.

## Cost
4 × 5 hr = 20 hr wall-clock (serial), ~10 hr if 2-way parallel.

## Inputs
- `agent_cli.py --provider claude-code` (no --schedule-only)
- Hot-start: `results/seed_optimizer.py`

## Outputs
- `results_agent_claude_fullopt_run{2..5}/`
- Run 1 = the existing `results_agent_claude_fullopt` (relabeled run1
  in aggregate.py).

## Success criteria
- Across 5 runs, no run rediscovers SLSQP from scratch (i.e., Claude's
  preference for SGD wrappers is reproducible).
- Best ROWP across 5 runs: report best/median/worst alongside the
  Gemini full-opt 5-seed numbers in Table 1. The two should be visually
  comparable in the paper table.

## Failure modes
- One run discovers SLSQP: the paper's "Claude takes a different path"
  framing must be softened to "Claude often takes a different path."
  This is a useful negative result and should be reported as-is.
- Quota: 5 × 5hr Claude usage may be expensive. Confirm budget.

## Launch
```
bash experiments/C_claude_fullopt_seeds/launch.sh
```

## Aggregation
```
pixi run python experiments/C_claude_fullopt_seeds/aggregate.py
```
