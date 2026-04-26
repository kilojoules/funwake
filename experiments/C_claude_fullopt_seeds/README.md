# Experiment C — Full-optimizer multi-seed (Claude or Codex)

## Reviewer concern
"Of the four (agent × interface) cells, only Gemini × full-optimizer has
n=5 agent-run seeds. Claude × full-optimizer has n=1. The 2×2 narrative
asymmetry is conspicuous."

## Hypothesis under test
The chosen agent's full-optimizer behavior (wrap topfarm_sgd_solve / no
SLSQP rediscovery for Claude; codex's pattern is unknown a priori) is
reproducible across agent runs.

## Method
Re-run the chosen frontier agent's full-optimizer loop 4 more times.
`AGENT=claude` (default) uses the Claude Code CLI; `AGENT=codex` uses
the OpenAI Codex CLI. Same hot-start, same 4.5 hr time budget per run.
Non-determinism comes from API temperature and timing.

## Cost
4 × 4.5 hr = 18 hr serial wall-clock per agent. Run claude OR codex
this week; defer the other if budget tight.

## Inputs
- `agent_cli.py --provider claude-code` (or `--provider codex --model gpt-5-codex`)
- Hot-start: `results/seed_optimizer.py`

## Outputs
- `results_agent_claude_fullopt_run{2..5}/` (or `_codex_fullopt_run{2..5}/`)
- Run 1 for claude = existing `results_agent_claude_fullopt`.

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
bash experiments/C_claude_fullopt_seeds/launch.sh                     # claude default
AGENT=codex bash experiments/C_claude_fullopt_seeds/launch.sh         # codex
```

## Aggregation
```
pixi run python experiments/C_claude_fullopt_seeds/aggregate.py
```
