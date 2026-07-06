# Flagship Run — Archived Evidence (Claude Code schedule-only search)

Complete retained record of the deployed Claude Code schedule-discovery run
that produced the dual-bump schedule (`iter_192`). Archived because the CLI's
default ~30-day transcript retention would otherwise delete it; this is the
authoritative primary record of what the agent saw and did.

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

## Memory-channel audit (resolves the "context artifacts" question)

Claude Code auto-memory was **enabled** (default config; runner sets no
`CLAUDE_CODE_DISABLE_AUTO_MEMORY`, no `--bare`). Empirically, it was **inert**:

1. The project memory store `~/.claude/projects/-work-users-juqu-funwake/memory/`
   was **empty** during the run (created 2026-04-18 23:14, 0 files, empty
   `MEMORY.md`). See `transcripts/MEMORY_DIR_LISTING.txt`.
2. Grep of all 49 retained transcripts for `Recalled memory` / `Writing memory`
   events: **zero**. The only "memory" strings are the agent discussing
   *limited-memory* quasi-Newton methods (L-BFGS-B), not CLI memory recall.

Conclusion: no memory-recall or memory-write occurred; the only in-context
information reaching the agent was the documented file artifacts (project-config
`CLAUDE.md`, the per-iteration user prompt, and the funwake runner's own
`agent_memory.md` scaffolding). Gemini CLI has no analogous feature.

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
