# Experiment A — Multi-agent-seed schedule-only

## Reviewer concern
"Of the four (agent × interface) cells, only Gemini × full-optimizer has n=5
agent-run seeds. Schedule-only is n=1. The headline mean ± std reported in
Table 1 measures *initialization* variance, not agent-run variance."

## Hypothesis under test
The schedule structures discovered by Claude and Gemini in schedule-only mode
are reproducible across independent agent runs (i.e. the agent did not just
get lucky on a single 5-hour loop).

## Method
Re-run each schedule-only agent loop 4 more times with different launch
RNG seeds. The "agent seed" controls only the launch (re-running the same
agent is non-deterministic by API design — temperature, retry behavior,
network timing all vary even with no seed). We record what the agent
discovered each time and compare structures (qualitatively) and best AEP
(quantitatively).

## Cost
8 runs × 5 hr = 40 hr wall-clock. Can run 2 in parallel locally (CPU-bound
agent loops, evaluation-bound). Total real time ≈ 20 hr if 2-way parallel,
40 hr serial.

## Inputs
- `agent_cli.py --provider claude-code --schedule-only`
- `agent_cli.py --provider gemini-cli --schedule-only`
- Hot-start: `results/seed_schedule.py` (existing schedule-only seed)

## Outputs
- `results_agent_claude_sched_run{1..5}/` — 5 Claude replicates
- `results_agent_gemini_sched_run{1..5}/` — 5 Gemini replicates
  (run 1 is the existing `results_agent_schedule_only_5hr` /
  `results_agent_gemini_cli_5hr`; symlink them in.)
- `attempt_log.json` per run with per-attempt train_aep + ROWP.

## Success criteria
- Across 5 replicates per agent, the BEST schedule's ROWP score has
  std < 5 GWh. (If higher, the discovery is fragile and should be
  reframed.)
- The structural motif (dual bumps for Claude, cosine restarts for
  Gemini) appears in ≥3 of 5 runs. (If <3, structure is one-off.)

## Failure modes & mitigations
- API quota exhaustion mid-run: see `launch.sh` resume logic.
- One agent crashes: skip; n=4 still informative.
- Discovered structures vary wildly: this is the result, write it up.

## Launch
```
bash experiments/A_multi_agent_seed/launch.sh
```

## Aggregation
```
pixi run python experiments/A_multi_agent_seed/aggregate.py
```
Writes `experiments/A_multi_agent_seed/summary.json` with per-run best
schedules, ROWP scores, and structural classification.
