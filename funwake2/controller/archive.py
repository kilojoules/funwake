"""MAP-elites archive on the FROZEN behavioral bins (spec 3.4, D-8).

Mirrors openevolve.database.ProgramDatabase's per-island MAP-Elites feature
grid (funwake2/vendor/openevolve/openevolve/database.py) but is self-contained
(no OpenEvolve runtime deps) and restricted to our four frozen descriptor
dimensions. One elite per (island, feature-cell); a candidate replaces the
incumbent only if it is feasible AND strictly fitter. Fully JSON-serializable
for atomic checkpoint/resume.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .descriptors import bin_descriptors, cell_label


@dataclass
class Entry:
    candidate_id: str
    source: str
    descriptors: Dict
    coord: Tuple            # (peak_bin, term_bin, coupling, restart_bin)
    fitness: float          # mean% over baseline (cross-cell aggregate)
    worst_cell: float       # min_c score_c (tiebreak)
    feasible: bool
    generation: int
    island: int
    parent_ids: List[str] = field(default_factory=list)
    engine: str = ""
    # firewall-safe per-cell feedback (%-scores + feasibility booleans only) so a
    # sampled parent can pass its cascade results to the mutator prompt.
    per_cell: Dict = field(default_factory=dict)

    def better_than(self, other: "Entry") -> bool:
        if other is None:
            return True
        # feasibility dominates, then mean fitness, then worst-cell tiebreak
        if self.feasible != other.feasible:
            return self.feasible
        if abs(self.fitness - other.fitness) > 1e-9:
            return self.fitness > other.fitness
        return self.worst_cell > other.worst_cell


class MapElitesArchive:
    def __init__(self, num_islands: int = 2):
        self.num_islands = num_islands
        # island -> {coord_key: Entry}
        self.islands: List[Dict[str, Entry]] = [dict() for _ in range(num_islands)]

    @staticmethod
    def _coord_key(coord) -> str:
        return "|".join(map(str, coord))

    def add(self, *, candidate_id, source, descriptors, fitness, worst_cell,
            feasible, generation, island, parent_ids=None, engine="",
            per_cell=None) -> Tuple[bool, Tuple]:
        """Attempt to place a candidate. Returns (accepted, coord)."""
        coord = bin_descriptors(descriptors)
        entry = Entry(candidate_id, source, descriptors, coord, fitness,
                      worst_cell, feasible, generation, island,
                      parent_ids or [], engine, per_cell or {})
        island = island % self.num_islands
        key = self._coord_key(coord)
        incumbent = self.islands[island].get(key)
        if entry.better_than(incumbent):
            self.islands[island][key] = entry
            return True, coord
        return False, coord

    def occupied_cells(self, island: Optional[int] = None) -> List[Tuple]:
        cells = set()
        rng = range(self.num_islands) if island is None else [island]
        for i in rng:
            for e in self.islands[i].values():
                cells.add(e.coord)
        return sorted(cells, key=lambda c: str(c))

    def all_entries(self, island: Optional[int] = None) -> List[Entry]:
        out = []
        rng = range(self.num_islands) if island is None else [island]
        for i in rng:
            out.extend(self.islands[i].values())
        return out

    def elites(self, island: Optional[int] = None) -> List[Entry]:
        return [e for e in self.all_entries(island) if e.feasible]

    def best(self) -> Optional[Entry]:
        feas = self.elites()
        if not feas:
            return None
        return max(feas, key=lambda e: (e.fitness, e.worst_cell))

    def summary(self) -> Dict:
        return {
            "num_islands": self.num_islands,
            "occupied_per_island": [len(m) for m in self.islands],
            "total_occupied_cells": len(self.occupied_cells()),
            "cells": [cell_label(c) for c in self.occupied_cells()],
        }

    # ── serialization (atomic checkpoint) ─────────────────────────────
    def to_dict(self) -> Dict:
        return {
            "num_islands": self.num_islands,
            "islands": [
                {k: asdict(e) for k, e in m.items()} for m in self.islands
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MapElitesArchive":
        arc = cls(d["num_islands"])
        for i, m in enumerate(d["islands"]):
            for k, ed in m.items():
                ed = dict(ed)
                ed["coord"] = tuple(ed["coord"])
                arc.islands[i][k] = Entry(**ed)
        return arc

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, sort_keys=True)   # deterministic bytes
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "MapElitesArchive":
        with open(path) as f:
            return cls.from_dict(json.load(f))
