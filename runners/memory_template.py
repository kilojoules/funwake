"""Portable agent memory template (Option 2: semi-structured).

This defines the standard agent_memory.md format used across all runners
(vLLM, Gemini, Claude Code, opencode). The format is:

1. **Status** — auto-populated by the runner after each eval
2. **Top Scripts** — auto-populated, ranked by feasible AEP
3. **Key Findings** — written by the agent, preserved across updates
4. **Next Experiments** — written by the agent, preserved across updates

Sections 1-2 are regenerated from attempt_log.json each turn.
Sections 3-4 are agent-authored and persisted across memory refreshes.

This is "Condition A" in the scaffolding comparison experiment.
"Condition B" (Claude Code native) uses Claude Code's built-in
conversation context and auto-memory instead of this file.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


def render_memory(
    attempt_log: list[dict],
    baseline_aep: float,
    time_budget_s: float,
    elapsed_s: float,
    agent_notes: str = "",
    top_n: int = 10,
) -> str:
    """Render agent_memory.md from structured data + agent's free-form notes.

    Args:
        attempt_log: List of attempt dicts from attempt_log.json.
        baseline_aep: Baseline AEP in GWh for gap computation.
        time_budget_s: Total time budget in seconds.
        elapsed_s: Elapsed time in seconds.
        agent_notes: Agent-authored markdown (Key Findings + Next Experiments).
                     Preserved verbatim across memory refreshes.
        top_n: Number of top scripts to show.

    Returns:
        Complete agent_memory.md content.
    """
    remaining_s = max(0, time_budget_s - elapsed_s)
    n_success = sum(1 for a in attempt_log if "train_aep" in a)
    n_error = sum(1 for a in attempt_log if "error" in a)

    # Top scripts: feasible first, then by AEP descending
    scored = [a for a in attempt_log if "train_aep" in a]
    scored.sort(key=lambda a: (
        not a.get("train_feasible", False),  # feasible first
        -a.get("train_aep", 0),              # then by AEP desc
    ))
    top = scored[:top_n]

    best_aep = top[0]["train_aep"] if top else 0.0
    best_feas = top[0].get("train_feasible", False) if top else False
    best_iter = top[0].get("attempt", "?") if top else "?"
    gap = best_aep - baseline_aep

    # ── Section 1: Status (auto-populated) ────────────────────────
    status = f"""## Status
- Elapsed: {elapsed_s/60:.1f} min / {time_budget_s/60:.0f} min ({remaining_s/60:.1f} min remaining)
- Attempts: {len(attempt_log)} ({n_success} scored, {n_error} errors)
- Baseline: {baseline_aep:.2f} GWh
- Best: {best_aep:.2f} GWh (#{best_iter}, {'feasible' if best_feas else 'INFEASIBLE'}, gap: {gap:+.2f})
"""

    # ── Section 2: Top Scripts (auto-populated) ───────────────────
    rows = ["## Top Scripts", "| # | AEP (GWh) | Feasible | Gap | Strategy |",
            "|---|-----------|----------|-----|----------|"]
    for a in top:
        feas = "Yes" if a.get("train_feasible") else "No"
        g = a.get("train_aep", 0) - baseline_aep
        strat = a.get("strategy", "?")
        rows.append(f"| {a.get('attempt', '?')} | {a.get('train_aep', 0):.2f} | "
                    f"{feas} | {g:+.2f} | {strat} |")
    top_scripts = "\n".join(rows) + "\n"

    # ── Sections 3-4: Agent-authored (preserved) ──────────────────
    if not agent_notes.strip():
        agent_notes = """## Key Findings
*(No findings yet — update this section after each eval.)*

## Next Experiments
- [ ] *(Add ideas here as you learn what works.)*
"""

    return f"""# Agent Memory

{status}
{top_scripts}
{agent_notes}"""


def extract_agent_notes(memory_md: str) -> str:
    """Extract the agent-authored sections from an existing agent_memory.md.

    Returns everything from '## Key Findings' onward, preserving the
    agent's free-form notes across memory refreshes.
    """
    # Find the start of agent-authored content
    for marker in ["## Key Findings", "## Findings", "## Lessons"]:
        idx = memory_md.find(marker)
        if idx >= 0:
            return memory_md[idx:].strip()
    return ""


def refresh_memory(
    memory_path: str,
    attempt_log: list[dict],
    baseline_aep: float,
    time_budget_s: float,
    elapsed_s: float,
    top_n: int = 10,
) -> str:
    """Read existing memory, preserve agent notes, regenerate status + top scripts.

    This is the main entry point for runners. Call after each eval.
    """
    existing = ""
    p = Path(memory_path)
    if p.exists():
        existing = p.read_text()

    agent_notes = extract_agent_notes(existing)

    content = render_memory(
        attempt_log=attempt_log,
        baseline_aep=baseline_aep,
        time_budget_s=time_budget_s,
        elapsed_s=elapsed_s,
        agent_notes=agent_notes,
        top_n=top_n,
    )

    p.write_text(content)
    return content
