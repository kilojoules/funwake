#!/usr/bin/env python
"""Real-engine 2-mutation SMOKE (Phase-3 pre-launch, round-2 gate).

Runs a live mutation engine (Claude Agent-SDK by default) inside the SCOPED
clean-room workspace and verifies: (1) each child is a parseable module defining
schedule_fn, (2) lineage + token/cost are logged, (3) the scope AND the full
transcript (prompt sent + children returned) are clean of forbidden path/holdout
tokens (`workspace.scan_tree`). Claude-first (item 3); pass --engine gemini once a
supported Gemini CLI/auth is restored.

Auth: Claude needs `pip install claude-agent-sdk` + `claude setup-token`
(CLAUDE_CODE_OAUTH_TOKEN); ANTHROPIC_API_KEY must be ABSENT (preflight refuses it).
This runner does NO evaluation (no jax) — it only exercises the mutation path.

Usage:
  pixi run python funwake2/controller/run_smoke.py --engine claude --n 2
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_FW2 = os.path.dirname(_THIS)
_ROOT = os.path.dirname(_FW2)
for _p in (_ROOT, _FW2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from funwake2.controller import workspace as W                 # noqa: E402
from funwake2.controller.engines.base import EvoContext        # noqa: E402


def _child_ok(src: str):
    if not src or not src.strip():
        return False, "empty child"
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"unparseable: {e}"
    has = any(isinstance(n, ast.FunctionDef) and n.name == "schedule_fn"
              for n in ast.walk(tree))
    return (has, "ok" if has else "no schedule_fn defined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--scope", default=os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "fw2_smoke_scope"))
    ap.add_argument("--out", default="funwake2/state/smoke")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # firewalled feedback (%-scores + booleans only — no AEP)
    feedback = {"dei_n50": {"score_pct": 0.08, "feasible": True},
                "parque_n20": {"score_pct": -0.02, "feasible": True},
                "parque_n14_uniform": {"score_pct": 0.0, "feasible": True}}
    parent_source = open(os.path.join(_FW2, "seeds", "native.py")).read()

    # scoped clean-room (materialize + assert_clean: tokens + ast.parse)
    scope = W.materialize(args.scope, parent_source=parent_source,
                          feedback=feedback, fw2_root=_FW2)
    print(f"[smoke] scoped workspace materialized + assert_clean OK: {scope}")

    if args.engine == "claude":
        from funwake2.controller.engines.claude_sdk import ClaudeSDKEngine, _build_prompt
        eng = ClaudeSDKEngine(cwd=scope)
    else:
        from funwake2.controller.engines.gemini_cli import GeminiCLIEngine, _build_prompt
        eng = GeminiCLIEngine(cwd=scope)
    try:
        eng.preflight()      # RAISES with a clear message if auth is missing
    except Exception as e:
        print(f"[smoke] {eng.name} preflight FAILED — auth not present yet:\n  {e}")
        print("[smoke] the scope + machinery are ready; re-run once auth lands.")
        sys.exit(2)
    print(f"[smoke] {eng.name} preflight OK (cwd-scoped)")

    ctx = EvoContext(parent_source=parent_source, parent_id="native_gen0",
                     per_cell_fitness=feedback, generation=0, island=0, notes="")
    prompt = _build_prompt(ctx, "") if args.engine == "claude" else _build_prompt(ctx)

    transcript_parts = [f"=== PROMPT ===\n{prompt}\n"]
    lineage, ok_count = [], 0
    for i in range(args.n):
        t0 = time.time()
        res = eng.mutate(ctx)
        child = res.source or ""
        ok, why = _child_ok(child)
        ok_count += int(ok)
        transcript_parts.append(f"=== CHILD {i} (ok={ok}: {why}) ===\n{child}\n")
        lg = res.log
        rec = {"i": i, "engine": lg.engine, "model": lg.model,
               "prompt_tokens": lg.prompt_tokens, "completion_tokens": lg.completion_tokens,
               "usd": lg.usd, "walltime_s": round(time.time() - t0, 2),
               "child_ok": ok, "why": why, "child_chars": len(child)}
        lineage.append(rec)
        print(f"[smoke] mutation {i}: ok={ok} ({why}) model={lg.model} "
              f"tok={lg.prompt_tokens}+{lg.completion_tokens} ${lg.usd}")

    # write lineage + transcript, then scan BOTH the scope and the transcript
    with open(os.path.join(args.out, f"lineage_{args.engine}.jsonl"), "w") as f:
        for r in lineage:
            f.write(json.dumps(r) + "\n")
    tpath = os.path.join(args.out, f"transcript_{args.engine}.txt")
    with open(tpath, "w") as f:
        f.write("\n".join(transcript_parts))

    scope_hits = W.scan_tree(scope)
    transcript_hits = W.scan_tree(tpath)
    clean = not scope_hits and not transcript_hits
    print(f"[smoke] scope forbidden-token hits: {scope_hits or 'NONE'}")
    print(f"[smoke] transcript forbidden-token hits: {transcript_hits or 'NONE'}")
    print(f"[smoke] children valid: {ok_count}/{args.n}  |  transcript clean: {clean}")
    summary = {"engine": args.engine, "n": args.n, "children_ok": ok_count,
               "transcript_clean": clean, "scope_hits": scope_hits,
               "transcript_hits": transcript_hits, "lineage": lineage,
               "total_usd": round(sum(r["usd"] for r in lineage), 6)}
    json.dump(summary, open(os.path.join(args.out, f"summary_{args.engine}.json"), "w"),
              indent=2)
    print(f"[smoke] RESULT: {'PASS' if (ok_count == args.n and clean) else 'FAIL'} "
          f"(children_ok={ok_count}/{args.n}, transcript_clean={clean})")
    sys.exit(0 if (ok_count == args.n and clean) else 1)


if __name__ == "__main__":
    main()
