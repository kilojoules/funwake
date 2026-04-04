"""Agent memory scaffolding — session persistence, transcript compaction,
and structured history for the FunWake optimization agent.

Inspired by claw-code's session_store, transcript, and history patterns.
Provides durable state across claude -p invocations and within long
Gemini sessions.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


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


# ── Agent Memory File ───────────────────────────────────────────────────

def render_agent_memory(state: SessionState, history: HistoryLog,
                        attempt_log: list[dict]) -> str:
    """Render agent_memory.md — the file Claude Code reads each iteration.

    Combines session state, recent history, attempt summary, and
    phase-specific guidance into a single markdown document.
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
