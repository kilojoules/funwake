"""Cascade evaluator (spec 3.2) wrapping funwake2.evaluator.evaluate.

Three stages, feasibility-gated at gamma_min at EVERY stage; each (schedule,
cell, seed, gamma_min, steps) result is content-addressed in the shared cache so
resume never recomputes:

  Stage A (fast-reject) — 2 cheap cells x 2 seeds. Reject if any seed is
    infeasible OR its AEP is below the per-cell c*D baseline by more than the
    noise floor. NOT the n200 high-N cell.
  Stage B (full)        — the full frozen training matrix x >=5 PAIRED seeds
    (same seed => same init as the baseline arm, G8 bit-stable). Per cell:
    score_c = 100 * (mean_seeds AEP_cand - mean_seeds AEP_baseline) / baseline,
    over the SAME seed subset (paired). Hard feasibility gate: any infeasible
    paired seed in a cell fails that cell => candidate fails. Aggregate:
    fitness = mean_c(score_c), worst = min_c(score_c) (worst-cell tiebreak).
  Stage C (elites only) — ROWP holdout margin (AEP FIREWALLED: only feasibility
    booleans + a margin-over-floor boolean cross the firewall) + the
    gamma_min = 1.0 responsiveness check (a schedule invariant to the tolerance
    is rejected).

Paired-seed baselines come from the frozen G2 table (controller/baselines_g2.json);
a missing (cell,seed) baseline is computed on the fly with the native seed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config as C
from .cache import ResultCache, result_key, schedule_hash

_THIS = os.path.dirname(os.path.abspath(__file__))
_FW2 = os.path.dirname(_THIS)          # funwake2/
_ROOT = os.path.dirname(_FW2)

BASELINE_PATH = os.path.join(_THIS, "baselines_g2.json")
NATIVE_SEED = os.path.join(_FW2, "seeds", "native.py")


def _is_feasibility_only(cell: str) -> bool:
    """True if the cell is registered feasibility-only (saturated objective ->
    hard feasibility gate retained, excluded from the mean-% score aggregate).
    Defensive: returns False if evaluator/CELLS is unavailable (fake-eval/dry)."""
    try:
        if _FW2 not in sys.path:
            sys.path.insert(0, _FW2)
        import evaluator as ev
        return bool(ev.CELLS.get(cell, {}).get("feasibility_only", False))
    except Exception:
        return False


# ── schedule loading (from a source string, cached by hash) ───────────
def _load_schedule_fn(source: str):
    fd, path = tempfile.mkstemp(suffix=".py", prefix="sched_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(source)
        spec = importlib.util.spec_from_file_location("cand_sched", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.schedule_fn
    finally:
        os.unlink(path)


def _load_native_source():
    with open(NATIVE_SEED) as f:
        return f.read()


def _load_native_fn():
    """Import funwake2/seeds/native.py by its REAL path so its ``__file__``-based
    pixwake path resolution works (a temp-file exec would break the import)."""
    spec = importlib.util.spec_from_file_location("native_ref", NATIVE_SEED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def load_baselines() -> Dict:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {"cells": {}}


@dataclass
class StageResult:
    stage: str
    passed: bool
    per_cell: Dict = field(default_factory=dict)   # cell -> {score,feasible,cand_mean,...}
    fitness: float = float("-inf")
    worst_cell: float = float("-inf")
    n_evals: int = 0
    notes: str = ""
    causes: Dict = field(default_factory=dict)     # stage-A rejection tally by cause


def _cell_farm(cell: str) -> str:
    """Farm a cell belongs to (for the farm-balanced aggregate). Prefer an
    explicit CELLS['farm'] tag; otherwise derive from the cell-name prefix
    (dei_* / parque_* / rowp_*)."""
    try:
        if _FW2 not in sys.path:
            sys.path.insert(0, _FW2)
        import evaluator as ev
        f = ev.CELLS.get(cell, {}).get("farm")
        if f:
            return f
    except Exception:
        pass
    return cell.split("_", 1)[0]


def _is_gbar_only(cell: str) -> bool:
    """True if the cell is a gbar-only (capability-frontier / stage-B+) cell.
    Off gbar it is PENDING (deferred to the elite tier) and NEVER gates."""
    try:
        if _FW2 not in sys.path:
            sys.path.insert(0, _FW2)
        import evaluator as ev
        return bool(ev.CELLS.get(cell, {}).get("gbar_only", False))
    except Exception:
        return False


class Cascade:
    def __init__(self, cfg: C.RunConfig, cache: ResultCache = None,
                 evaluate_fn=None, baselines=None):
        self.cfg = cfg
        self.cache = cache or ResultCache(cfg.cache_dir)
        self.baselines = baselines if baselines is not None else load_baselines()
        self._native_source = None
        self._native_fn = None
        # evaluate_fn injectable for tests; defaults to the real jax evaluator
        self._evaluate = evaluate_fn

    # ── the single evaluation primitive (cached) ──────────────────────
    def _eval(self, sched_hash: str, schedule_fn, cell: str, seed: int,
              gamma_min: float, steps: int) -> Dict:
        key = result_key(sched_hash, cell, seed, gamma_min, steps)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        rec = self._evaluate_impl(cell, schedule_fn, seed, steps, gamma_min)
        self.cache.put(key, rec)
        return rec

    def _evaluate_impl(self, cell, schedule_fn, seed, steps, gamma_min) -> Dict:
        if self._evaluate is not None:
            return self._evaluate(cell, schedule_fn, seed, steps, gamma_min)
        # lazy import the jax evaluator only when a real eval is needed
        if _FW2 not in sys.path:
            sys.path.insert(0, _FW2)
        import evaluator as ev
        return ev.evaluate(cell, schedule_fn, seed=seed,
                           total_steps=steps, gamma_min=gamma_min)

    # ── baseline arm (paired) — AEP_ref is a SCALE CONSTANT ONLY ──────
    def _baseline_aep(self, cell: str, seed: int, gamma_min: float, steps: int):
        """Return the c*D native reference AEP for (cell, seed).

        BLOCKING fitness patch (coordinator): AEP_ref is used PURELY as a scale
        constant to make cells of different magnitude commensurable. It is
        returned REGARDLESS of whether the reference layout is feasible — the
        reference's feasibility is irrelevant to normalization; only the
        CANDIDATE's own feasibility at gamma_min gates (see _score_cell). This
        matters for capability-frontier cells (e.g. n200) whose native reference
        may itself be infeasible.
        """
        cb = self.baselines.get("cells", {}).get(cell)
        if cb and str(seed) in cb.get("seeds", {}):
            return cb["seeds"][str(seed)]
        # compute native reference on the fly (paired), cached
        if self._native_source is None:
            self._native_source = _load_native_source()
        if self._native_fn is None:
            self._native_fn = _load_native_fn()     # imported by real path
        nh = schedule_hash(self._native_source)
        rec = self._eval(nh + "_baseline", self._native_fn, cell, seed, gamma_min, steps)
        return rec.get("aep_gwh")     # scale constant; feasibility ignored here

    # ── per-cell paired score ─────────────────────────────────────────
    def _score_cell(self, sched_hash, schedule_fn, cell, seeds, gamma_min, steps):
        cand_aeps, feas_all, n = [], True, 0
        for s in seeds:
            rec = self._eval(sched_hash, schedule_fn, cell, s, gamma_min, steps)
            n += 1
            if "error" in rec:
                return {"cell": cell, "feasible": False, "score": float("-inf"),
                        "error": rec["error"][:120], "n_evals": n}
            cand_aeps.append(rec["aep_gwh"])
            feas_all = feas_all and bool(rec.get("feasible"))   # CANDIDATE gate
        base = [self._baseline_aep(cell, s, gamma_min, steps) for s in seeds]
        cand_mean = statistics.fmean(cand_aeps)
        base_mean = statistics.fmean(base)   # scale constant (may be infeasible ref)
        score = 100.0 * (cand_mean - base_mean) / base_mean if base_mean else float("-inf")
        # HARD GATE is the candidate's own feasibility, independent of the ref.
        return {"cell": cell, "feasible": feas_all,
                "score": score if feas_all else float("-inf"),
                "cand_mean": round(cand_mean, 4), "base_mean": round(base_mean, 4),
                "n_evals": n}

    # ── STAGE A (GROSS fast-reject) ───────────────────────────────────
    def stage_a(self, source: str, cells: List[str], seeds: List[int]) -> StageResult:
        """Cheap gross filter — reject only clearly-bad candidates so the archive
        keeps the quality-diversity exploration it exists for. A (cell, seed) is
        rejected iff the candidate is INFEASIBLE at gamma_min OR its AEP is more
        than ``stage_a_reject_frac`` (a gross ~1%) BELOW the reference. This is
        deliberately NOT texture-floor-tight (a floor-tight Stage A would
        mass-reject exploratory but viable schedules). The texture floors are used
        for selection margins, not here. ``causes`` tallies the rejection reason."""
        h = schedule_hash(source)
        fn = _load_schedule_fn(source)
        per_cell, n, passed = {}, 0, True
        causes = {"ok": 0, "infeasible": 0, "below_ref": 0, "error": 0}
        gm, steps = self.cfg.gamma_min, self.cfg.total_steps
        frac = self.cfg.stage_a_reject_frac
        for cell in cells:
            for s in seeds:
                rec = self._eval(h, fn, cell, s, gm, steps)
                n += 1
                if "error" in rec:
                    passed = False; causes["error"] += 1
                    per_cell.setdefault(cell, {})[s] = {"cause": "error"}; continue
                base = self._baseline_aep(cell, s, gm, steps)
                feas = bool(rec.get("feasible"))
                below = base and (rec["aep_gwh"] < base * (1.0 - frac))
                cause = "infeasible" if not feas else ("below_ref" if below else "ok")
                causes[cause] += 1
                ok = (cause == "ok")
                per_cell.setdefault(cell, {})[s] = {
                    "feasible": feas, "aep": rec["aep_gwh"], "base": round(base, 3),
                    "ok": ok, "cause": cause}
                passed = passed and ok
        return StageResult("A", passed, per_cell, n_evals=n, causes=causes)

    # ── STAGE B ───────────────────────────────────────────────────────
    def stage_b(self, source: str, cells: List[str], seeds: List[int]) -> StageResult:
        h = schedule_hash(source)
        fn = _load_schedule_fn(source)
        gm, steps = self.cfg.gamma_min, self.cfg.total_steps
        per_cell, n, feasible_all = {}, 0, True
        scored = []          # (farm, score) for the farm-balanced aggregate
        all_scores = []      # every scored cell's score (worst-cell tiebreak)
        for cell in cells:
            # PENDING: gbar-only capability-frontier cells are deferred off gbar
            # (enable_stage_b_plus=False). They never gate and never score here.
            if _is_gbar_only(cell) and not self.cfg.enable_stage_b_plus:
                per_cell[cell] = {"cell": cell, "status": "PENDING",
                                  "deferred_to": "stage_b_plus", "gates": False,
                                  "n_evals": 0}
                continue
            fo = _is_feasibility_only(cell)
            # feasibility-only cells run at 2 seeds (they gate feasibility only)
            cell_seeds = list(seeds[:2]) if fo else list(seeds)
            sc = self._score_cell(h, fn, cell, cell_seeds, gm, steps)
            sc["feasibility_only"] = fo
            per_cell[cell] = sc
            n += sc["n_evals"]
            # HARD GATE (all scored + feasibility-only cells): candidate must be
            # feasible in EVERY such cell.
            feasible_all = feasible_all and sc["feasible"]
            # SCORE AGGREGATE excludes feasibility-only cells (saturated objective).
            if not fo:
                scored.append((_cell_farm(cell), sc["score"]))
                all_scores.append(sc["score"])
        if not feasible_all or not scored:
            return StageResult("B", False, per_cell, n_evals=n,
                               notes="infeasible cell (hard gate)")
        # FARM-BALANCED aggregate: mean over farms of the per-farm mean cell score,
        # so each farm contributes equally regardless of how many cells it has
        # (the training set has more DEI cells than Parque cells).
        by_farm: Dict = {}
        for farm, s in scored:
            by_farm.setdefault(farm, []).append(s)
        farm_means = [statistics.fmean(v) for v in by_farm.values()]
        fitness = statistics.fmean(farm_means)
        worst = min(all_scores)     # worst-cell tiebreak (over scored cells)
        return StageResult("B", True, per_cell, fitness=fitness,
                           worst_cell=worst, n_evals=n,
                           notes=f"farm-balanced over {sorted(by_farm)}")

    # ── STAGE B+ (elite-tier, gbar ONLY) ──────────────────────────────
    def stage_b_plus(self, elites: List[tuple]) -> Dict:
        """Re-score the top-k archive elites on expensive high-N cells (incl.
        n200) with 2-3 paired seeds. **gbar ONLY** — refuses to run unless
        cfg.enable_stage_b_plus is True, so it NEVER executes inside the Mac
        evolution loop. `elites` is a list of (candidate_id, source).

        Same fitness convention as stage B (AEP_ref = scale constant; candidate
        feasibility is the hard gate). Capability-frontier cells whose native
        reference is infeasible still yield a well-defined %-score.
        """
        if not self.cfg.enable_stage_b_plus:
            raise RuntimeError(
                "stage_b_plus is gbar-only (enable_stage_b_plus=False). It must "
                "NOT run in the Mac evolution loop — n200-class evals are ~5-6 min.")
        cells = list(self.cfg.stage_b_plus_cells)
        seeds = self.cfg.stage_b_plus_seeds
        gm, steps = self.cfg.gamma_min, self.cfg.total_steps
        out = {}
        for cid, source in elites[: self.cfg.stage_b_plus_top_k]:
            h = schedule_hash(source)
            fn = _load_schedule_fn(source)
            per_cell = {}
            for cell in cells:
                per_cell[cell] = self._score_cell(h, fn, cell, seeds, gm, steps)
            out[cid] = per_cell
        return out

    # ── STAGE C (elites) — FIREWALLED ─────────────────────────────────
    def stage_c(self, source: str, holdout_cell: str, seeds: List[int],
                floor_gwh: float, resp_cell: Optional[str] = None) -> Dict:
        """Returns a FIREWALL-SAFE summary: feasibility booleans, margin-over-
        floor boolean, responsiveness boolean, cell id. The raw holdout AEP and
        margin GWh are kept in `._firewalled` and NEVER returned to any prompt.
        """
        h = schedule_hash(source)
        fn = _load_schedule_fn(source)
        gm, steps = self.cfg.gamma_min, self.cfg.total_steps
        cand, feas = [], True
        for s in seeds:
            rec = self._eval(h, fn, holdout_cell, s, gm, steps)
            if "error" in rec:
                return {"stage": "C", "cell": holdout_cell, "feasible": False,
                        "error": rec["error"][:120]}
            cand.append(rec["aep_gwh"]); feas = feas and bool(rec.get("feasible"))
        base = [self._baseline_aep(holdout_cell, s, gm, steps) for s in seeds]
        margin = statistics.fmean(cand) - statistics.fmean(base)   # FIREWALLED

        # gamma_min responsiveness: re-score at 1.0 m, confirm behavior changes
        resp_cell = resp_cell or holdout_cell
        r001 = self._eval(h, fn, resp_cell, seeds[0], 0.01, steps)
        r1 = self._eval(h, fn, resp_cell, seeds[0], 1.0, steps)
        responds = (("error" not in r001 and "error" not in r1)
                    and (r001.get("feasible") != r1.get("feasible")
                         or abs(r001.get("aep_gwh", 0) - r1.get("aep_gwh", 0)) > floor_gwh))

        firewalled = {"holdout_cell": holdout_cell, "margin_gwh": round(margin, 4),
                      "cand_mean": round(statistics.fmean(cand), 4)}
        return {
            "stage": "C", "cell": holdout_cell, "feasible": feas,
            "margin_over_floor": bool(feas and margin > floor_gwh),
            "responds_to_gamma_min": bool(responds),
            "floor_gwh": floor_gwh,
            "_firewalled": firewalled,          # stripped before any prompt
        }
