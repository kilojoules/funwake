# Flagship Run — Archived Evidence (Gemini CLI schedule-only search)

Complete retained record of the deployed Gemini CLI schedule-discovery run,
the counterpart to the Claude run in `../flagship_run_sophia/`. Together they
document both deployed agents.

## Provenance

- **Host**: local workstation (macOS), user `julianquick`
- **Working dir**: `/Users/julianquick/funwake` (Gemini project alias `funwake`,
  projectHash `03eee786ae7d…`, per `~/.gemini/projects.json`)
- **Dates**: 2026-04-07 → 2026-04-08 (`iter_192.py` mtime 2026-04-08 02:15)
- **Agent**: `gemini -p` headless invocations, one per search iteration, driven
  by `runners/gemini_cli_runner.py`.
- **Model / CLI version**: single model `gemini-3-flash-preview` throughout
  (transcript `model` field). Gemini CLI `@google/gemini-cli@0.47.0` — the
  *current install*; the run-time version is **not separately logged** (session
  JSON has no version field).

## Memory / context audit — the Claude↔Gemini asymmetry

Gemini CLI has **no automatic memory-recall feature** (unlike Claude Code). Its
only context channels are deterministic and were empty/absent here:

1. **No global `~/.gemini/GEMINI.md`** — nothing injected across sessions.
2. **`~/.gemini/settings.json`** contains only the oauth type — no
   checkpointing, savedChats, or memory configuration enabled
   (`gemini_settings.json`).
3. Session `.json` under `~/.gemini/tmp/funwake/chats/` are per-invocation
   checkpoints; they are **not auto-reloaded** into later invocations (Gemini
   resumes a saved chat only on explicit `/chat resume`, which the headless
   runner never issued).

So Gemini's per-iteration context was exactly the documented artifacts: the
project `GEMINI.md` and the per-iteration prompt, plus the funwake runner's own
`agent_memory.md` scaffolding (`run_record/agent_memory.md`). This confirms the
paper's asymmetry statement: Claude Code had an auto-memory *mechanism* (enabled
but empirically empty/inert — see the sophia manifest), Gemini had **no such
mechanism at all**.

## Contents

- `transcripts/` — 63 session `.json` (2026-04-07/08) = the raw Gemini agent I/O,
  plus `logs.json`.
- `run_record/` — `attempt_log.json`, `iter_192.py` (deployed), and
  `agent_memory.md` (the runner scaffolding). Full 186 iterations live in the
  repo at `results_agent_gemini_cli_5hr/`.
- `gemini_settings.json` — proof no memory/checkpoint features were enabled.

## Not included / notes

- A **separate later** Gemini run exists (`~/.gemini/tmp/funwake-1`, 2026-04-14→19,
  277 sessions, 46 MB) — NOT the deployed flagship; omitted.
- `~/.gemini/oauth_creds.json` deliberately **excluded** (credential). Sessions
  secret-scanned (OAuth tokens, API keys, PEM) — none found.
