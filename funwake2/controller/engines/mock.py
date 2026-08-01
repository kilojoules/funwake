"""MOCK mutation engine — machinery validation ONLY, ZERO LLM spend.

Applies a deterministic small edit to the parent schedule (perturb one numeric
literal, chosen by a hash of (parent, generation, index)) and reports synthetic
token/cost so the cost tracker and lineage log exercise their full paths.
Deterministic: the same (parent, gen, idx) always yields the same child +
synthetic cost, which is what lets the checkpoint->kill->resume test reproduce
bit-identically. NO network, NO credentials, NO provider is contacted.
"""
from __future__ import annotations

import hashlib
import re
import time

from .base import Engine, EvoContext, MutationLog, MutationResult


class MockEngine(Engine):
    name = "mock"

    def __init__(self, model: str = "mock-mutator-v1", usd_per_call: float = 0.01,
                 tokens_per_call: int = 1500):
        self.model = model
        self.usd_per_call = usd_per_call
        self.tokens_per_call = tokens_per_call

    def preflight(self) -> None:
        return None   # nothing to check; never contacts a provider

    def _seed(self, ctx: EvoContext) -> int:
        # deterministic in (parent, gen, island, child_index) ONLY — no global
        # counter — so a resumed run regenerates identical children.
        raw = f"{ctx.parent_id}|{ctx.generation}|{ctx.island}|{ctx.child_index}"
        return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)

    def mutate(self, ctx: EvoContext) -> MutationResult:
        t0 = time.time()
        h = self._seed(ctx)
        src = ctx.parent_source

        # find float literals and perturb exactly one, deterministically
        floats = list(re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", src))
        if floats:
            pick = floats[h % len(floats)]
            val = float(pick.group(1))
            factor = 0.90 + 0.20 * ((h >> 8) % 1000) / 1000.0    # in [0.90, 1.10)
            new_val = round(val * factor, 6)
            child = src[:pick.start()] + repr(new_val) + src[pick.end():]
            tag = f"perturb literal {pick.group(1)} -> {new_val}"
        else:
            child = src + f"\n# mock-noop mutation g{ctx.generation}\n"
            tag = "noop-append"

        # synthetic, deterministic cost (varies a little per call)
        pt = self.tokens_per_call + (h % 500)
        ct = 400 + (h % 300)
        usd = self.usd_per_call * (1.0 + (h % 100) / 1000.0)
        log = MutationLog(engine=self.name, model=self.model,
                          prompt_tokens=pt, completion_tokens=ct,
                          usd=round(usd, 6), walltime_s=round(time.time() - t0, 4),
                          ok=True, error=tag)
        return MutationResult(source=child, log=log)
