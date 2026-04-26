# Week-2 Experiments — Reviewer-Driven Strengthening

Each subdirectory contains one experiment that addresses a specific reviewer
concern. **Nothing in this directory should be launched without first reading
its README and the master prelaunch checklist below.**

| Exp | Addresses                                         | Cost (hr) | Status     | Eng prereq |
|-----|---------------------------------------------------|-----------|------------|------------|
| A   | "Single agent run per cell"                       | 24–40     | drafted, codex agent wired | none |
| B   | "Scaffold confound (CLI vs structured-output)"    | 10        | drafted    | new runner: `runners/anthropic_api_runner.py` + `agent_cli.py` provider choice |
| C   | "Full-opt asymmetry across agents"                | 12–20     | drafted, supports claude or codex | none |
| D   | "Uniform-wind failure: systematic or seed-noise?" | 30 (CPU)  | drafted    | none       |
| E   | "Random-search family is post-hoc"                | 3         | drafted, smoke-tested | family.py renders + executes |
| F   | "Single held-out polygon"                         | 10        | drafted, polygon-builder smoke-tested | none |
| G   | "Discovery is model-specific?"                    | 3.5–4.5   | drafted, codex default | none |

Three frontier *agentic CLIs* are now wired in:
`claude-code` (Anthropic), `gemini-cli` (Google), `codex` (OpenAI).
Plus open-source paths via `vllm` and `opencode`. The narrow-vs-broad
interface comparison in the paper now spans 3 frontier providers (A, C, G).

`drafted` = scripts exist, no run executed.
`launched` = run started (note SLURM JID / local PID in the subdir status file).
`integrated` = numbers/figures merged into `paper/short.tex`.

## Prelaunch checklist (apply per experiment)

1. Read the experiment's `README.md`. Confirm the addressed concern still
   needs addressing (paper may have moved on).
2. Confirm output dir does not collide with prior runs.
3. For experiments that use commercial APIs (A, B, C, G): verify
   `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are set, check current quota.
4. For LUMI experiments (D matrix baselines, optionally A/C if rerouted):
   verify project budget on `project_465002609`.
5. Update this file's status column.
6. Commit the launch script and prelaunch checklist results before the run.
   Reproducibility starts with knowing exactly what was launched.

## Aggregation

After all targeted experiments complete, run

    pixi run python experiments/aggregate.py

to refresh `paper/figs/*` and produce a `experiments/results_summary.json`
with the new numbers. The aggregator does NOT auto-edit `short.tex` —
inspect the summary and integrate by hand.

## Pre-registration note

Experiment E is *pre-registered*: the schedule family is defined and
checked-in BEFORE running the random search. Once checked in, do not
modify `family.py` based on results. If a follow-up family is needed,
create `family_v3.py` and disclose the change. The locked design lives
in `E_preregistered_random_search/PREREGISTRATION.md`.

## Suggested week-2 schedule

| Day | Action                                                                   | Wall hours |
|-----|--------------------------------------------------------------------------|------------|
| 1   | Launch A (4 Claude + 4 Gemini sched, 2-way parallel)                     | 20         |
| 1   | Launch C (4 Claude full-opt) on a separate machine if available          | parallel   |
| 1   | Launch E (320 samples, sequential locally)                               | 3          |
| 2   | Implement `runners/anthropic_api_runner.py` for B                        | 1 day eng  |
| 2-3 | Launch B (sched + full-opt under structured-output)                      | 10         |
| 3   | Launch D on LUMI as 20-shard array job                                   | 1 (queue)  |
| 4   | Launch F (build polygon → 500-start baseline → score top schedules)      | 7          |
| 4   | Launch G (Claude Opus or alternative)                                    | 5          |
| 5-6 | Aggregate, regenerate figures, update paper, write a "what we learned"   | —          |
| 7   | Final compile, anonymization sweep, fonts check, submit                  | —          |
