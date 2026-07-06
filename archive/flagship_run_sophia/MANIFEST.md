# Claude — sophia archive (CORRECTED SCOPE)

> **Correction (2026-07-06).** This directory was first labeled the flagship
> schedule-only search. That is wrong. The `transcripts/` here are a **different
> experiment** — full-optimizer discovery + LUMI/vast.ai benchmarking, run
> **2026-04-18/19** on git branch `lumi-vastai-benchmark` (first prompt: "read
> the seed optimizer … full workflow"; 0/49 contain the schedule-designer
> prompt, 41/49 mention lumi/vast/benchmark). The **deployed dual-bump search
> ran 2026-04-05, locally**, and its transcripts fall outside the CLI's ~30-day
> retention (unrecoverable).
>
> What is authoritative here: **`run_record/`** — the `attempt_log.json` of the
> real deployed search (320 attempts, all 2026-04-05, ~5 h; attempt 192 = the
> dual-bump at 13:09) and `iter_192.py`. The `transcripts/` are retained as a
> record of the *separate* Apr-18 full-opt/benchmark run, not the deployed
> schedule search.
>
> **Model of the dual-bump: not determinable** — see `../MODELS.md`.

## Provenance

- **Host**: `sophia.dtu.dk` (`sophia1.hpc.ait.dtu.dk`), user `juqu`
- **Working dir**: `/work/users/juqu/funwake`
- **Config dir**: default `~/.claude` (no `CLAUDE_CONFIG_DIR`, no isolation)
- **Dates**: 2026-04-18 → 2026-04-19 (`iter_192.py` mtime 2026-04-18 23:12).
  NOTE: the paper draft (Appendix A) says 2026-04-05/04-07 — that predates this
  run; reconcile before submission. The 04-05 date matches an earlier *local*
  copy, not this deployed HPC run.
- **Agent**: `claude -p` headless invocations, one per search iteration, driven
  by `runners/claude_code_runner.py` (see `harness/`).

## Memory-channel audit (RE-SCOPED)

This audit was performed on the transcripts in this directory — which are the
**Apr-18 full-opt/benchmark sessions, not the deployed Apr-5 schedule search**.
Two facts still stand for *these* sessions, and are relevant context:

1. Those sessions ran **Claude Code v2.1.0** (transcript `version` field), which
   **predates auto-memory** (shipped v2.1.59) — so the feature did not exist in
   them regardless of config.
2. The project memory store `~/.claude/projects/-work-users-juqu-funwake/memory/`
   was empty (`transcripts/MEMORY_DIR_LISTING.txt`), and no
   `Recalled memory` / `Writing memory` events appear.

**For the deployed Apr-5 schedule search, memory behavior is NOT directly
verifiable** — its transcripts are unretained. What bounds it: the funwake
project memory store is empty; the harness disables nothing
(`claude_code_runner.py` sets no `CLAUDE_CODE_DISABLE_AUTO_MEMORY`, no `--bare`);
and a content audit of the only candidate stores (the `~/clusters` project:
researcher profile + HPC notes) found nothing referencing the test farm, its
AEP, or any schedule — the schedule-related memory postdates deployment by ~2
months. So: no task-relevant memory could have leaked, but "zero recall events"
cannot be asserted for the deployed run. Gemini CLI has no analogous feature
(see `../flagship_run_gemini/MANIFEST.md`).

## Contents

- `transcripts/` — 49 session `.jsonl` (main sessions + Task subagents), the raw
  agent I/O. `MEMORY_DIR_LISTING.txt` = proof the memory store was empty.
- `run_record/` — `attempt_log.json` (full search log) + kept iterations
  (`iter_153`, `iter_192` [deployed], `iter_262`).
- `harness/` — `agent_cli.py`, `runners/`, `models.json`, `CLAUDE.md`, `run.sh`,
  and the run dir's `.claude/` config, as they were at run time.

## Collection

Pulled from sophia on 2026-07-06 via `tar` of the two locations above; secret-
scanned (`sk-ant-`, `ANTHROPIC_API_KEY`, Google/GitHub tokens, PEM) — none found.
Transcripts contain absolute paths and the user email (research record, not
secrets).
