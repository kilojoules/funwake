"""Append-only lineage / provenance log (spec 4 provenance requirements).

One JSONL record per candidate, fsync'd per record. Every field the
pre-registration demands for a "the system discovered X" claim:

    content hash, parent ID(s), mutation engine + model string,
    prompt/completion tokens, $ cost, wall-time, behavioral descriptors,
    per-cell fitness, generation, timestamp.

Seeded ancestors (native, iter192/181/118 ports) are logged as generation 0
with their port transforms so novelty is measured against what was seeded.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional


class LineageLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # touch so resume can read even before the first record
        if not os.path.exists(path):
            open(path, "a").close()
        # dedupe key set (candidate_id, generation) so a re-run generation on
        # resume does NOT double-log — keeps the JSONL bit-identical.
        self._seen = set()
        for rec in self.read_all():
            self._seen.add((rec.get("candidate_id"), rec.get("generation")))

    def append(self, record: Dict) -> None:
        record.setdefault("timestamp", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_candidate(self, *, candidate_id: str, parent_ids: List[str],
                      engine: str, model: str, prompt_tokens: int,
                      completion_tokens: int, usd: float, walltime_s: float,
                      descriptors: Dict, per_cell_fitness: Dict,
                      generation: int, island: int,
                      port_transform: Optional[str] = None,
                      ancestor: Optional[str] = None,
                      stage_reached: str = "", status: str = "") -> None:
        dk = (candidate_id, generation)
        if dk in self._seen:
            return          # already logged (resume re-run) — idempotent
        self._seen.add(dk)
        self.append({
            "candidate_id": candidate_id,
            "parent_ids": parent_ids,
            "engine": engine,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "usd": round(float(usd), 6),
            "walltime_s": round(float(walltime_s), 2),
            "descriptors": descriptors,
            "per_cell_fitness": per_cell_fitness,
            "generation": generation,
            "island": island,
            "ancestor": ancestor,
            "port_transform": port_transform,
            "stage_reached": stage_reached,
            "status": status,
        })

    def read_all(self) -> List[Dict]:
        out = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        out.append(json.loads(ln))
        return out
