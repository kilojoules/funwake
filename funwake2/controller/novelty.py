"""ShinkaEvolve grafts (spec 3.1 / D-5): two eval-saving mechanisms ported as
~100-line additions on top of the OpenEvolve chassis.

  1. code-novelty rejection-sampling  — reject a proposal whose code is a
     near-duplicate of something already evaluated, BEFORE spending a (costly)
     stage-B eval. Mirrors ShinkaEvolve's embedding + cheap-LLM dedup; here the
     "embedding" is a token-shingle set and similarity is Jaccard (the cheap
     first pass). A real run additionally routes borderline cases to a cheap-LLM
     judge (openevolve.novelty_judge) — wired via `llm_judge` hook, unused/None
     in the dry run so NO LLM is called.
  2. fitness/novelty-aware parent sampling — sample a parent weighted by a blend
     of fitness rank and behavioral novelty (distance to the archive), so the
     search neither collapses onto one elite nor wanders uniformly.

References: SakanaAI/ShinkaEvolve (arXiv 2509.19349); see funwake2/vendor/PIN.md.
"""
from __future__ import annotations

import random
import re
from typing import Callable, List, Optional

from .cache import schedule_hash


def _shingles(source: str, k: int = 5) -> set:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+\.?[0-9]*|\S", source)
    toks = [t for t in toks if t]
    if len(toks) < k:
        return set(toks)
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def code_similarity(a: str, b: str) -> float:
    """Jaccard over k-shingles (the cheap novelty 'embedding')."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


class NoveltyFilter:
    """Rejection-sampling gate: reject near-duplicate code before stage B."""

    def __init__(self, threshold: float = 0.92,
                 llm_judge: Optional[Callable[[str, str], bool]] = None):
        self.threshold = threshold
        self.seen_sources: List[str] = []
        self.seen_hashes: set = set()
        self.llm_judge = llm_judge   # None in dry run -> NO LLM call
        self.n_rejected = 0
        self.n_accepted = 0

    def register(self, source: str) -> None:
        h = schedule_hash(source)
        if h not in self.seen_hashes:
            self.seen_hashes.add(h)
            self.seen_sources.append(source)

    # ── checkpoint/resume: persist the seen-source set so a resumed run makes
    #    the SAME novelty rejections (required for bit-identical resume) ──────
    def state(self) -> dict:
        return {"sources": list(self.seen_sources)}

    def load_state(self, d: dict) -> None:
        for src in d.get("sources", []):
            self.register(src)

    def is_novel(self, source: str) -> bool:
        h = schedule_hash(source)
        if h in self.seen_hashes:
            self.n_rejected += 1
            return False   # exact dup
        for prev in self.seen_sources:
            sim = code_similarity(source, prev)
            if sim >= self.threshold:
                # cheap-LLM confirmation stage (skipped/None in dry run)
                if self.llm_judge is None or self.llm_judge(source, prev):
                    self.n_rejected += 1
                    return False
        self.n_accepted += 1
        return True


def novelty_aware_parent(archive, rng: random.Random, novelty_weight: float = 0.5):
    """Sample a parent Entry weighted by fitness rank blended with behavioral
    novelty (how sparsely its feature-cell neighborhood is populated)."""
    entries = archive.all_entries()
    if not entries:
        return None
    # fitness component (rank-normalized, feasible favored). Sort by the content
    # hash so the sampling order is CANONICAL — identical whether the archive is
    # in insertion order or was reloaded from a (sorted) checkpoint. This is what
    # makes a resumed generation pick the same parents => bit-identical lineage.
    feas = [e for e in entries if e.feasible] or entries
    feas = sorted(feas, key=lambda e: e.candidate_id)
    order = sorted(feas, key=lambda e: (e.fitness, e.worst_cell))
    fit_rank = {id(e): (i + 1) / len(order) for i, e in enumerate(order)}
    # novelty component: inverse local density of its feature-cell
    from collections import Counter
    dens = Counter(e.coord for e in feas)
    weights = []
    for e in feas:
        f = fit_rank[id(e)]
        nov = 1.0 / dens[e.coord]
        weights.append((1 - novelty_weight) * f + novelty_weight * nov)
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    return rng.choices(feas, weights=weights, k=1)[0]
