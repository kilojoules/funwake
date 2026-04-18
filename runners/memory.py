"""Agent memory scaffolding — session persistence, transcript compaction,
and structured history for the FunWake optimization agent.

Inspired by claw-code's session_store, transcript, and history patterns.
Provides durable state across claude -p invocations and within long
Gemini sessions.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .strategy_taxonomy import (
    SCHEDULE_FAMILIES, FULLOPT_FAMILIES, classify,
    all_family_names, describe_family, family_by_name,
)


# ── History Log ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HistoryEvent:
    """A single notable event during the agent session."""
    timestamp: float
    kind: str          # "attempt", "error", "phase", "discovery", "milestone"
    title: str
    detail: str


@dataclass
class HistoryLog:
    """Structured log of agent events, persisted as markdown."""
    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, kind: str, title: str, detail: str = "") -> None:
        self.events.append(HistoryEvent(
            timestamp=time.time(), kind=kind,
            title=title, detail=detail,
        ))

    def as_markdown(self) -> str:
        lines = ["# Session History", ""]
        for e in self.events:
            ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            lines.append(f"- [{ts}] **{e.kind}**: {e.title}")
            if e.detail:
                lines.append(f"  {e.detail}")
        return "\n".join(lines)

    def recent(self, n: int = 10) -> list[HistoryEvent]:
        return self.events[-n:]


# ── Transcript Store ────────────────────────────────────────────────────

@dataclass
class TranscriptEntry:
    """One turn of the conversation."""
    role: str          # "user", "assistant", "tool_call", "tool_result"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None


@dataclass
class TranscriptStore:
    """Compactable conversation transcript.

    Keeps the full transcript in memory but can compact old entries
    into a summary, preserving recent context for the LLM.
    """
    entries: list[TranscriptEntry] = field(default_factory=list)
    compaction_count: int = 0

    def append(self, role: str, content: str, tool_name: str = None) -> None:
        self.entries.append(TranscriptEntry(
            role=role, content=content, tool_name=tool_name,
        ))

    def compact(self, keep_last: int = 20) -> str:
        """Compress old entries into a summary, return the summary text."""
        if len(self.entries) <= keep_last:
            return ""

        old = self.entries[:-keep_last]
        self.entries = self.entries[-keep_last:]
        self.compaction_count += 1

        # Build summary of compacted entries
        tool_calls = [e for e in old if e.tool_name]
        tool_summary = {}
        for e in tool_calls:
            tool_summary[e.tool_name] = tool_summary.get(e.tool_name, 0) + 1

        summary = (
            f"[Transcript compacted (#{self.compaction_count}). "
            f"Removed {len(old)} entries. "
            f"Tool calls: {dict(tool_summary)}.]"
        )
        return summary

    def token_estimate(self) -> int:
        """Rough token count (4 chars per token)."""
        return sum(len(e.content) for e in self.entries) // 4


# ── Session Store ───────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Durable session state persisted between invocations."""
    session_id: str
    start_time: float
    time_budget: float
    best_aep: float = 0.0
    best_iter: int = 0
    best_strategy: str = ""
    baseline_aep: float = 0.0
    attempts_total: int = 0
    attempts_success: int = 0
    attempts_error: int = 0
    strategies_tried: list[str] = field(default_factory=list)
    phase: str = "explore"     # "explore" or "exploit"
    consecutive_sgd_solve: int = 0
    discoveries: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def remaining(self) -> float:
        return max(0, self.time_budget - self.elapsed())

    def in_phase2(self, fraction: float = 0.3) -> bool:
        return self.elapsed() / self.time_budget > fraction


def save_session(state: SessionState, path: Path) -> None:
    """Persist session state to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    data["_saved_at"] = time.time()
    path.write_text(json.dumps(data, indent=2))


def load_session(path: Path) -> Optional[SessionState]:
    """Load session state from JSON, or None if not found."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data.pop("_saved_at", None)
    return SessionState(**data)


# ── Strategy Registry ───────────────────────────────────────────────────

def _classify_attempt_dir(output_dir: str, mode: str) -> dict[str, list[int]]:
    """Walk all iter_*.py files in output_dir and classify each into families.

    Returns {family_name: [attempt_numbers]} of the families observed.
    """
    observed: dict[str, list[int]] = {}
    pattern = os.path.join(output_dir, "iter_*.py")
    for path in sorted(glob.glob(pattern)):
        try:
            m = os.path.basename(path)[5:-3]  # iter_NNN.py → NNN
            attempt_num = int(m)
        except ValueError:
            continue
        try:
            with open(path) as f:
                src = f.read()
        except OSError:
            continue
        for fam in classify(src, mode):
            observed.setdefault(fam, []).append(attempt_num)
    return observed


def _scan_sibling_runs(this_dir: str, mode: str) -> dict[str, list[str]]:
    """Scan all results_agent_* dirs (same taxonomy mode) and collect their
    observed families. Excludes this_dir itself. This gives a cross-run
    knowledge pool so a fresh seed sees what families prior runs already
    explored.

    `mode` = "schedule" → include dirs that appear to be schedule-only runs
    `mode` = "fullopt"  → include dirs that appear to be full-optimizer runs

    Returns {family_name: [sibling_dir_names_where_tried]}.
    """
    this_dir = os.path.abspath(this_dir)
    parent = os.path.dirname(this_dir)

    all_run_dirs = [
        d for d in glob.glob(os.path.join(parent, "results_agent_*"))
        if os.path.isdir(d) and os.path.abspath(d) != this_dir
    ]

    # Same-mode filter: schedule dirs have 'sched' or 'schedule' in the name
    sibling_dirs = []
    for d in all_run_dirs:
        base = os.path.basename(d)
        is_sched = "sched" in base or "schedule" in base
        if mode == "schedule" and is_sched:
            sibling_dirs.append(d)
        elif mode == "fullopt" and not is_sched:
            sibling_dirs.append(d)

    cross: dict[str, list[str]] = {}
    for sib in sibling_dirs:
        observed = _classify_attempt_dir(sib, mode)
        for fam in observed:
            cross.setdefault(fam, []).append(os.path.basename(sib))
    return cross


def _best_feasible_for_family(observed: dict[str, list[int]],
                              attempts: list[dict]) -> dict[str, float]:
    """For each observed family, return the best feasible TRAIN AEP across
    attempts assigned to it. Families with no feasible attempts are omitted.
    """
    by_attempt = {a.get("attempt"): a for a in attempts if "attempt" in a}
    best: dict[str, float] = {}
    for fam, attempt_nums in observed.items():
        feasible_aeps = []
        for n in attempt_nums:
            a = by_attempt.get(n)
            if not a:
                continue
            if a.get("train_feasible") is False:
                continue
            aep = a.get("train_aep")
            if aep is not None:
                feasible_aeps.append(aep)
        if feasible_aeps:
            best[fam] = max(feasible_aeps)
    return best


def render_strategy_registry(
    output_dir: str,
    attempts: list[dict],
    mode: str = "schedule",
    close_after_n: int = 3,
) -> str:
    """Render the UNEXPLORED / TRIED / CLOSED strategy registry.

    `mode` selects taxonomy ("schedule" or "fullopt").
    `close_after_n` marks a family CLOSED once >= N attempts in it
    have been tried in this seed or found in sibling seeds.
    """
    observed = _classify_attempt_dir(output_dir, mode)
    best_per_family = _best_feasible_for_family(observed, attempts)
    cross = _scan_sibling_runs(output_dir, mode)

    all_names = all_family_names(mode)
    tried_local = set(observed.keys()) - {"uncategorized"}
    tried_any_seed = tried_local | set(cross.keys()) - {"uncategorized"}
    unexplored = [n for n in all_names if n not in tried_any_seed]

    closed_local = {
        fam for fam, nums in observed.items()
        if len(nums) >= close_after_n and fam != "uncategorized"
    }
    closed_cross = {
        fam for fam, sibs in cross.items()
        if len(sibs) >= 2 and fam != "uncategorized"  # tried in ≥2 sibling seeds
    }
    closed = closed_local | closed_cross

    # Partially-explored = tried but not yet closed
    partial = (tried_local | set(cross.keys())) - closed - {"uncategorized"}

    lines: list[str] = []
    lines.append("## Strategy Registry")
    lines.append("")
    lines.append(f"*Mode: {mode}. Close-after: {close_after_n} attempts.*")
    lines.append("")

    # UNEXPLORED keeps full descriptions so the agent knows what each
    # family is. CLOSED and PARTIAL are listed by name only — the agent
    # has already seen those descriptions, no need to re-render every turn.
    lines.append(f"### UNEXPLORED — try these FIRST ({len(unexplored)} remaining)")
    if unexplored:
        for name in unexplored:
            lines.append(f"- [ ] **{name}** — {describe_family(name, mode)}")
    else:
        lines.append("- *(all taxonomy families tried at least once — free exploration mode)*")
    lines.append("")

    if partial:
        names = []
        for name in sorted(partial):
            best = best_per_family.get(name)
            mark = f" ({best:.1f})" if best is not None else ""
            names.append(f"{name}{mark}")
        lines.append(f"### PARTIALLY EXPLORED ({len(partial)}): "
                     + ", ".join(names))
        lines.append("")

    if closed:
        names = []
        for name in sorted(closed):
            best = best_per_family.get(name)
            mark = f" ({best:.1f})" if best is not None else ""
            names.append(f"{name}{mark}")
        lines.append(f"### CLOSED ({len(closed)}, do not revisit): "
                     + ", ".join(names))
        lines.append("")

    # Mandatory next action
    if unexplored:
        next_fam = unexplored[0]
        lines.append("### MANDATORY NEXT ACTION")
        lines.append("")
        lines.append(
            f"The UNEXPLORED list is non-empty. Your next attempt MUST "
            f"introduce a family from that list. Do NOT write another "
            f"variant of a family already in PARTIALLY EXPLORED or CLOSED.")
        lines.append("")
        lines.append(f"**Suggested starting point:** `{next_fam}` — "
                     f"{describe_family(next_fam, mode)}")
        lines.append("")
        lines.append("Before writing code, state in your response:")
        lines.append("1. Which UNEXPLORED family you chose.")
        lines.append("2. Why that family is a plausible fit for "
                     "non-convex wake-interaction optimization with "
                     "polygon constraints.")
        lines.append("3. What observable result would make you mark it "
                     "CLOSED vs keep iterating.")
    else:
        lines.append("### MANDATORY NEXT ACTION")
        lines.append("")
        lines.append("All taxonomy families have been tried. Free exploration "
                     "mode: the most promising avenue is PARTIAL entries "
                     "with the highest best_train, or combinations across "
                     "families not yet tried together.")

    lines.append("")
    return "\n".join(lines)


# ── Agent Memory File ───────────────────────────────────────────────────

def render_agent_memory(state: SessionState, history: HistoryLog,
                        attempt_log: list[dict],
                        output_dir: Optional[str] = None,
                        mode: str = "schedule") -> str:
    """Render agent_memory.md — the file Claude Code reads each iteration.

    Combines session state, recent history, attempt summary, strategy
    registry, and phase-specific guidance into a single markdown document.

    If `output_dir` is provided, the Strategy Registry is computed by
    scanning iter_*.py files in that directory and in sibling seed dirs.
    """
    sections = []

    # Header
    sections.append(f"# Agent Memory — {time.strftime('%H:%M:%S')}")

    # Time budget
    sections.append(f"""
## Time Budget
- Elapsed: {state.elapsed()/60:.1f} min
- Remaining: {state.remaining()/60:.1f} min
- Budget: {state.time_budget/60:.0f} min total
- Phase: {state.phase}
""")

    # Performance
    gap = state.best_aep - state.baseline_aep
    sections.append(f"""
## Performance
- Baseline: {state.baseline_aep:.1f} GWh
- Best so far: {state.best_aep:.1f} GWh (attempt {state.best_iter}, gap: {gap:+.1f})
- Attempts: {state.attempts_total} ({state.attempts_success} success, {state.attempts_error} errors)
- Strategies tried: {', '.join(sorted(set(state.strategies_tried))) or 'none'}
""")

    # Strategy registry — computed from iter_*.py files in output_dir
    if output_dir:
        try:
            sections.append(render_strategy_registry(
                output_dir, attempt_log, mode=mode))
        except Exception as e:
            sections.append(f"## Strategy Registry\n*(registry render failed: {e})*\n")

    # Discoveries
    if state.discoveries:
        sections.append("## Discoveries")
        for d in state.discoveries:
            sections.append(f"- {d}")
        sections.append("")

    # Recent attempts
    recent = attempt_log[-10:] if attempt_log else []
    if recent:
        sections.append("## Recent Attempts")
        for a in recent:
            if "train_aep" in a:
                delta = a["train_aep"] - state.baseline_aep
                strat = a.get("strategy", "?")
                sections.append(
                    f"- #{a['attempt']}: AEP={a['train_aep']:.1f} "
                    f"({delta:+.1f}) [{strat}] "
                    f"t={a.get('train_time', '?'):.0f}s"
                )
            elif "error" in a:
                sections.append(f"- #{a['attempt']}: ERROR — {a['error'][:80]}")
        sections.append("")

    # Phase-specific guidance
    if state.in_phase2():
        sections.append("""## Phase 2: Exploration

You've been running for a while. Consider:
- Custom gradient descent with jax.grad (not topfarm_sgd_solve)
- Wind-direction-aware grid initialization
- Different penalty schedules (alpha should INCREASE as lr decays)
- Diverse multi-start with varied initialization strategies
""")

    if state.consecutive_sgd_solve >= 5:
        sections.append("""## Diversity Alert

Your last 5+ optimizers all wrap topfarm_sgd_solve. Try something
fundamentally different: custom Adam loop, simulated annealing,
evolutionary placement, or wind-aware initialization.
""")

    # Recent history events
    recent_events = history.recent(5)
    if recent_events:
        sections.append("## Recent Events")
        for e in recent_events:
            ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            sections.append(f"- [{ts}] {e.title}: {e.detail}")
        sections.append("")

    return "\n".join(sections)
