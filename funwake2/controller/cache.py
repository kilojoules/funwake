"""Content-addressed evaluation-result cache (spec 5.3).

Every eval result is a JSON keyed by (schedule_hash, cell, seed, gamma_min,
steps). On resume, a key already present is a cache HIT and is NOT recomputed —
a killed run resumes at the exact missing evals with zero recompute. Because the
skeleton_v2 G8 float32-canonicalization makes (schedule, cell, seed) bit-stable
across fresh processes, cached AEP is bit-identical to a fresh recompute.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional


def schedule_hash(source: str) -> str:
    """Stable content hash of a schedule module's source.

    Comments / blank lines / trailing whitespace are stripped so cosmetically
    different but semantically identical schedules dedupe to the same key.
    """
    lines = []
    for ln in source.splitlines():
        s = re.sub(r"#.*$", "", ln).rstrip()
        if s.strip():
            lines.append(s)
    norm = "\n".join(lines)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def result_key(sched_hash: str, cell: str, seed: int, gamma_min: float,
               steps: int) -> str:
    raw = f"{sched_hash}|{cell}|{seed}|{gamma_min:g}|{steps}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ResultCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if os.path.exists(p):
            self.hits += 1
            with open(p) as f:
                return json.load(f)
        self.misses += 1
        return None

    def put(self, key: str, result: dict) -> None:
        p = self._path(key)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)   # atomic
