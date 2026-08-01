"""Mutation-engine interface.

An engine turns a parent schedule (source) + evolution context into a child
schedule (source) and a MutationLog carrying the provenance the lineage log
requires: engine name, resolved model string, prompt/completion tokens, $ cost,
wall-time.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MutationLog:
    engine: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    usd: float
    walltime_s: float
    ok: bool = True
    error: str = ""

    @property
    def tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@dataclass
class MutationResult:
    source: Optional[str]
    log: MutationLog


@dataclass
class EvoContext:
    """What the prompt sampler hands the engine. Holdout/test AEP NEVER appears
    here (firewall) — only feasibility booleans + per-cell %-over-baseline for
    training cells."""
    parent_source: str
    parent_id: str
    generation: int
    island: int
    child_index: int = 0        # deterministic index within the generation (resume-safe)
    per_cell_fitness: Dict = field(default_factory=dict)
    inspirations: List[str] = field(default_factory=list)   # sibling elite sources
    notes: str = ""


class Engine(abc.ABC):
    name = "base"

    def preflight(self) -> None:
        """Raise if the engine is unsafe/misconfigured to start. Default: OK."""
        return None

    @abc.abstractmethod
    def mutate(self, ctx: EvoContext) -> MutationResult:
        ...
