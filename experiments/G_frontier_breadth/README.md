# Experiment G — Additional frontier model

## Reviewer concern (anticipated)
"Is the discovery effect Claude-/Gemini-specific, or does any frontier
model produce schedules that exceed baseline given the narrow
interface?"

## Hypothesis under test
A third frontier model (e.g., Claude Opus, GPT-class via API) given the
narrow interface produces a schedule that beats the 500-start baseline.

## Method
Run one additional frontier model under the schedule-only interface,
same hot-start, same time budget (5 hr).

Candidates (pick 1, ordered by integration cost):
1. **Codex CLI** (`gpt-5-codex` via OpenAI's `codex` binary) — drop-in
   via `runners/codex_runner.py`. Recommended default. Lets us compare
   three frontier *agentic CLIs* on the same task.
2. **Claude Opus 4.x** via the existing `claude-code` runner — drop-in,
   only the model alias changes.
3. **A larger open-weight model** (Llama 3.3 405B) — already supported
   by `vllm_runner.py` but requires LUMI GPU budget (~5–10 GPU-hr).

For week-2, default is option 1 (Codex). Run schedule-only first
(`MODE=sched`); then full-opt (`MODE=full`) if budget allows.

## Cost
1 × 3 hr agent run + scoring overhead ≈ 3.5 hr (sched) or 4.5 hr (full).

## Inputs
- `agent_cli.py --provider codex --model gpt-5-codex` (default)
- Hot-start: `results/seed_schedule.py` (sched) or `results/seed_optimizer.py` (full)

## Outputs
- `results_agent_gpt_5_codex_sched/`
- `results_agent_gpt_5_codex_full/` (if MODE=full)

## Success criteria
- Best feasible ROWP ≥ baseline + 10 GWh: confirms breadth.
- The discovered schedule structure: report whether it matches Claude's
  dual-bump pattern (model-family carry-over) or invents something new.

## Failure modes
- Opus is slower per turn → fewer iterations within the 5 hr budget.
  Mitigation: extend budget to 7.5 hr if early iteration count is low.

## Launch
```
bash experiments/G_frontier_breadth/launch.sh
```

## Aggregation
```
pixi run python experiments/G_frontier_breadth/aggregate.py
```
