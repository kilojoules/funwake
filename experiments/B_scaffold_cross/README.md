# Experiment B — Scaffold cross-experiment

## Reviewer concern
"Full-optimizer Claude/Gemini ran with a different scaffold (rich CLI agents
managing their own loops via `claude -p` / `gemini -p`) than the open-source
schedule-only runs (structured-output prompting via vLLM). The interface
(narrow vs broad) is therefore confounded with the scaffold (CLI vs SO).
The paper's central claim — that the *interface* drives discovery vs
reimplementation — is not yet causal."

## Hypothesis under test
Holding the agent and interface fixed, the scaffold does NOT change the
qualitative outcome (discovery vs reimplementation).

## The 2 × 2 × 2 grid (and what we already have)

|                 | scaffold = CLI (`claude -p`)       | scaffold = structured-output (vLLM JSON) |
|-----------------|------------------------------------|------------------------------------------|
| narrow × Claude | `results_agent_schedule_only_5hr`  | **B1 — to run**                          |
| narrow × Gemini | `results_agent_gemini_cli_5hr`     | (deprioritized — Gemini-flash via vLLM not configured) |
| broad × Claude  | `results_agent_claude_fullopt`     | **B2 — to run**                          |
| broad × Gemini  | `results_agent_gemini_fullopt_2hr` | (already covered by VLLMRunner with open-source Llama 3.3) |

Cells **B1** and **B2** are the diagonal: re-run Claude under
the *opposite* scaffold for both interfaces. If outcomes flip with scaffold,
the scaffold-not-interface alternative is supported. If outcomes match,
the interface claim survives.

## Method
We do NOT have a structured-output Claude path today. Approximate it with
two options (pick one based on engineering cost):

**Option B-a (cheap, partial):** Re-run Claude full-optimizer under
`agent_cli.py`'s structured-output runner pointed at Anthropic's API
*directly* (not via `claude -p`). Requires adding an "anthropic-api"
runner that uses tool-use protocol but does not invoke Claude Code's
agentic loop. ~1 day eng. ~5 hr compute.

**Option B-b (fair, expensive):** Run an open-weight model (Llama 3.3
70B) under both scaffolds — already have VLLMRunner; need to write
a Llama-via-CLI runner using `llamafile` or similar. ~3 days eng.
~10 hr compute.

For week-2, **B-a** is the planned path.

## Cost
1 cell × 2 interfaces × 5 hr = 10 hr compute, plus ~1 day eng to
implement the runner. Sequential, no parallelism needed.

## Inputs
- New runner: `runners/anthropic_api_runner.py` (to be written)
- Existing: `playground/skeleton.py`, `tools/run_optimizer.py`
- Hot-start: `results/seed_optimizer.py` (full) or `results/seed_schedule.py` (sched)

## Outputs
- `results_agent_claude_anthropic_api_sched/`
- `results_agent_claude_anthropic_api_fullopt/`

## Success criteria
A 2×2 contingency: which cells produced "discovery" (novel schedule motif)
vs "reimplementation" (SLSQP/SGD wrapper). If the diagonal flips relative
to the off-diagonal, scaffold matters. If not, interface matters.

## Failure modes
- Anthropic API tool-use response format mismatch with our harness:
  fall back to logging raw output and post-hoc parsing.
- Agent does not converge on either approach in 5 hr: extend to 10 hr
  before declaring inconclusive.

## Launch
```
bash experiments/B_scaffold_cross/launch.sh
```

## Aggregation
```
pixi run python experiments/B_scaffold_cross/aggregate.py
```
Writes a 2×2 contingency table.
