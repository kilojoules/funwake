# Model & CLI-version forensics (CORRECTED)

> **Correction (2026-07-06).** An earlier version of this file attributed the
> Claude dual-bump to Opus 4.5 (and, before that, a workflow guessed Opus 4.6).
> **Both are unsupported.** The transcripts archived under
> `flagship_run_sophia/transcripts/` are **NOT** the schedule-only search that
> produced the dual-bump — they are a *different* experiment (full-optimizer
> discovery + LUMI/vast.ai benchmarking, 2026-04-18/19, git branch
> `lumi-vastai-benchmark`). The actual schedule-search transcripts (2026-04-05,
> run locally) fall outside the CLI's ~30-day retention and are unrecoverable.

## What produced the deployed Claude dual-bump (`iter_192`)

- **Run**: schedule-only search, **2026-04-05**, 320 attempts, ~5 h (attempt 192
  = the dual-bump, timestamp 13:09, ~2.8 h into the run). Record:
  `flagship_run_sophia/run_record/attempt_log.json` (valid).
- **Model: not determinable.** Checks that could have pinned it, all negative:
  - `runners/claude_code_runner.py` invokes `claude -p … --output-format text`
    with **no `--model`** → Claude Code account default, not pinned; and no
    per-invocation model metadata captured.
  - Claude Code ran via **subscription** (`userType: external`), not an API key
    → no server-side per-model usage to query.
  - The generating transcript is not retained.
- **Candidates** available and in use across the project in this period:
  `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-opus-4-6`
  (all released before 2026-04-05; Opus 4.6 shipped 2026-02-05). We **cannot
  determine** which produced the deployed schedule.
- **Weak prior only** (not evidence): the account's *default* in the unrelated
  2026-04-18 sessions was `claude-opus-4-5-20251101` even though 4.6 had been out
  10 weeks — suggesting a tier/config default of 4.5 — but that is a different
  experiment on a different host two weeks later.

## Models in the archived (mislabeled) Apr-18 full-opt/benchmark sessions

For the record — these describe the 49 `flagship_run_sophia/transcripts/`, i.e.
the full-opt + benchmark run, **not** the dual-bump:

| count | model |
|--:|--|
| 1383 | `claude-opus-4-5-20251101` (primary + subagents) |
| 372 | `claude-sonnet-4-5-20250929` |
| 34 | `claude-opus-4-6` |
| 12 | `claude-haiku-4-5-20251001` |

Claude Code version (transcript `version` field): **`2.1.0`** (dominant), one
`2.1.78` record. Note `2.1.0` predates auto-memory (shipped v2.1.59), so those
sessions had no auto-memory feature regardless of config.

## Gemini deployed schedule — recoverable

Gemini's search transcripts (2026-04-07/08) **do** cover its deployed iteration
(attempt 192, 2026-04-08 02:12). Single model throughout: **`gemini-3-flash-preview`**
(660 messages, no mix). Gemini CLI **0.47.0** (`@google/gemini-cli`, current
install; the exact run-time version is not separately logged). The deployed
Gemini schedule is therefore attributable; the Claude one is not.

## Honest paper sentence

> The Gemini schedule was produced by `gemini-3-flash-preview` (Gemini CLI).
> The Claude dual-bump was produced via Claude Code under the account default
> model (the runner pinned none); the generating session's transcript is not
> retained and the model is not recoverable from run records. Models available
> and in use during the search period were `claude-opus-4-5-20251101`,
> `claude-sonnet-4-5-20250929`, and `claude-opus-4-6`.
