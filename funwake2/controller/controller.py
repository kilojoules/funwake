"""FunWake-2 evolutionary controller (our custom driver over the OpenEvolve-style
MAP-elites archive + cascade + ShinkaEvolve novelty grafts).

Responsibilities: seed generation 0 with the (self-contained, scale-aware)
ancestors, then run island generations — novelty-aware parent sampling ->
mutation engine -> code-novelty rejection -> cascade eval -> archive update ->
lineage log -> per-generation atomic checkpoint -> cost-ceiling abort at 90%.

Determinism / resume (spec 5.3): parent sampling RNG is seeded from
(run_id, generation), the mock engine is deterministic in (parent, gen, island,
child_index), and every eval is content-addressed in the shared cache. So a
generation re-run on resume reproduces identical candidates and hits the cache
for every eval (zero recompute); lineage dedupe keeps the JSONL bit-identical.
Checkpoints are at generation boundaries (atomic); `generation` in meta.json is
the next generation to run.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

from . import config as C
from .archive import MapElitesArchive
from .cache import ResultCache, schedule_hash
from .cascade import Cascade
from .cost import CostTracker
from .descriptors import compute_descriptors, bin_descriptors, cell_label
from .lineage import LineageLog
from .novelty import NoveltyFilter, novelty_aware_parent
from .engines.base import EvoContext
from .workspace import sanitize as _sanitize_src

_THIS = os.path.dirname(os.path.abspath(__file__))
_SEED_DIR = os.path.join(_THIS, "seeds")
GEN0_SEEDS = [
    ("seed_native", "seed_native.py", "native"),
    ("seed_cosine", "seed_cosine.py", "iter192/181-family"),
    ("seed_cyclic", "seed_cyclic.py", "iter118-family"),
]


def _read(path):
    with open(path) as f:
        return f.read()


class Controller:
    def __init__(self, cfg: C.RunConfig, engine, cascade: Cascade = None):
        self.cfg = cfg
        self.engine = engine
        self.cache = ResultCache(cfg.cache_dir)
        self.cascade = cascade or Cascade(cfg, cache=self.cache)
        self.lineage = LineageLog(cfg.lineage_path)
        self.novelty = NoveltyFilter(cfg.novelty_threshold)
        self.cost = CostTracker(cfg.max_usd, cfg.max_tokens, cfg.abort_fraction)
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.archive = MapElitesArchive(cfg.num_islands)
        self.generation = 0
        self.seeded = False
        self.done = False
        self.aborted = False
        self._load_state()

    # ── cells for the run (dry-run overrides) ─────────────────────────
    @property
    def stage_a_cells(self):
        if self.cfg.dry_run:
            return self.cfg.dry_cells or ["dei_n50", "parque_n20"]
        import evaluator as ev  # noqa
        return [k for k, v in ev.CELLS.items() if v.get("stage_a")]

    @property
    def stage_b_cells(self):
        if self.cfg.dry_run:
            return self.cfg.dry_cells or ["dei_n50", "parque_n20"]
        import evaluator as ev  # noqa
        return [k for k, v in ev.CELLS.items() if v.get("stage_b")]

    @property
    def holdout_cell(self):
        # real run: ROWP farm-level holdout. dry run: a cheap training cell
        # stands in so stage C fires in order without a slow ROWP eval.
        return "parque_n10_omnidir" if self.cfg.dry_run else "rowp_n74"

    # ── persistence ───────────────────────────────────────────────────
    def _meta_path(self):
        return os.path.join(self.cfg.state_dir, "meta.json")

    def _arc_path(self):
        return os.path.join(self.cfg.state_dir, "archive.json")

    def _nov_path(self):
        return os.path.join(self.cfg.state_dir, "novelty.json")

    def checkpoint(self):
        self.archive.save(self._arc_path())
        # persist novelty state so a resumed generation reproduces rejections
        ntmp = self._nov_path() + ".tmp"
        with open(ntmp, "w") as f:
            json.dump(self.novelty.state(), f)
            f.flush(); os.fsync(f.fileno())
        os.replace(ntmp, self._nov_path())
        meta = {
            "run_id": self.cfg.run_id,
            "generation": self.generation,
            "seeded": self.seeded,
            "done": self.done,
            "aborted": self.aborted,
            "cost": self.cost.to_dict(),
            "config": self.cfg.as_dict(),
        }
        tmp = self._meta_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self._meta_path())

    def _load_state(self):
        if os.path.exists(self._arc_path()):
            self.archive = MapElitesArchive.load(self._arc_path())
        if os.path.exists(self._nov_path()):
            with open(self._nov_path()) as f:
                self.novelty.load_state(json.load(f))
        if os.path.exists(self._meta_path()):
            with open(self._meta_path()) as f:
                meta = json.load(f)
            self.generation = meta.get("generation", 0)
            self.seeded = meta.get("seeded", False)
            self.done = meta.get("done", False)
            self.aborted = meta.get("aborted", False)
            self.cost = CostTracker.from_dict(meta["cost"])
            # The checkpointed cost is only as fresh as the last generation
            # boundary; a mid-generation crash can leave spend that the lineage
            # already recorded but meta.json does not. Reconcile UPWARD to the
            # lineage totals so the 90% ceiling is never silently exceeded.
            self._reconcile_cost_from_lineage()

    def _reconcile_cost_from_lineage(self):
        path = self.cfg.lineage_path
        if not os.path.exists(path):
            return
        usd = 0.0; tokens = 0; calls = 0
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    u = float(r.get("usd", 0) or 0)
                    pt = int(r.get("prompt_tokens", 0) or 0)
                    ct = int(r.get("completion_tokens", 0) or 0)
                    if u or pt or ct:
                        usd += u; tokens += pt + ct; calls += 1
        except Exception:
            return
        st = self.cost.state
        st.usd = max(st.usd, usd)
        st.tokens = max(st.tokens, tokens)
        st.n_calls = max(st.n_calls, calls)

    # ── gen-0 seeding ─────────────────────────────────────────────────
    def seed_gen0(self):
        for cid, fname, ancestor in GEN0_SEEDS:
            src = _read(os.path.join(_SEED_DIR, fname))
            h = schedule_hash(src)
            self.novelty.register(src)
            desc = compute_descriptors(_import_fn(os.path.join(_SEED_DIR, fname)))
            coord = bin_descriptors(desc)
            a = self.cascade.stage_a(src, self.stage_a_cells, self.cfg.stage_a_seeds)
            if a.passed:
                b = self.cascade.stage_b(src, self.stage_b_cells, self.cfg.stage_b_seeds)
                fitness, worst, feas = b.fitness, b.worst_cell, b.passed
                per_cell = _fw_percell(b.per_cell)
                stage = "B"
            else:
                fitness, worst, feas, per_cell, stage = float("-inf"), float("-inf"), False, {}, "A"
            self.archive.add(candidate_id=h, source=src, descriptors=desc,
                             fitness=fitness, worst_cell=worst, feasible=feas,
                             generation=0, island=0, parent_ids=[], engine="seed",
                             per_cell=per_cell)
            self.lineage.log_candidate(
                candidate_id=h, parent_ids=[], engine="seed", model="seed",
                prompt_tokens=0, completion_tokens=0, usd=0.0, walltime_s=0.0,
                descriptors=desc, per_cell_fitness=per_cell, generation=0,
                island=0, ancestor=ancestor, port_transform="self-contained scale-aware",
                stage_reached=stage, status=("elite" if feas else "infeasible"))
        self.seeded = True

    # ── one evolutionary generation ───────────────────────────────────
    def step_generation(self) -> str:
        g = self.generation
        rng = random.Random(f"{self.cfg.run_id}:{g}")
        for i in range(self.cfg.proposals_per_gen):
            if self.cost.should_abort():
                self.aborted = True
                self.checkpoint()
                return "ABORT"
            parent = novelty_aware_parent(self.archive, rng)
            if parent is None:
                break
            island = i % self.cfg.num_islands
            # FIREWALL: sanitize the parent source before it enters the prompt
            # (a gen-0 seed's docstring can carry a forbidden path token); and
            # pass the parent's firewall-safe per-cell fitness so the mutator
            # actually sees the cascade feedback it is meant to improve on.
            ctx = EvoContext(parent_source=_sanitize_src(parent.source),
                             parent_id=parent.candidate_id,
                             generation=g, island=island, child_index=i,
                             per_cell_fitness=dict(getattr(parent, "per_cell", {}) or {}),
                             notes="")
            res = self.engine.mutate(ctx)
            # cost accounting from the per-invocation engine log
            self.cost.add(res.log.usd, res.log.tokens)
            if res.source is None:
                self.lineage.log_candidate(
                    candidate_id=f"{parent.candidate_id}:g{g}:i{i}:fail",
                    parent_ids=[parent.candidate_id], engine=res.log.engine,
                    model=res.log.model, prompt_tokens=res.log.prompt_tokens,
                    completion_tokens=res.log.completion_tokens, usd=res.log.usd,
                    walltime_s=res.log.walltime_s, descriptors={}, per_cell_fitness={},
                    generation=g, island=island, stage_reached="mutate",
                    status="mutation-failed")
                continue
            child = res.source
            h = schedule_hash(child)
            # ShinkaEvolve code-novelty rejection BEFORE spending a stage-B eval
            if not self.novelty.is_novel(child):
                self.lineage.log_candidate(
                    candidate_id=h, parent_ids=[parent.candidate_id],
                    engine=res.log.engine, model=res.log.model,
                    prompt_tokens=res.log.prompt_tokens,
                    completion_tokens=res.log.completion_tokens, usd=res.log.usd,
                    walltime_s=res.log.walltime_s, descriptors={}, per_cell_fitness={},
                    generation=g, island=island, stage_reached="novelty",
                    status="rejected-duplicate")
                continue
            self.novelty.register(child)
            # A real LLM can return syntactically invalid Python or a module with
            # no schedule_fn; log it as failed and continue rather than crash the
            # whole run.
            try:
                child_fn = _fn_from_source(child)
                desc = compute_descriptors(child_fn)
            except Exception as e:
                self.lineage.log_candidate(
                    candidate_id=h, parent_ids=[parent.candidate_id],
                    engine=res.log.engine, model=res.log.model,
                    prompt_tokens=res.log.prompt_tokens,
                    completion_tokens=res.log.completion_tokens, usd=res.log.usd,
                    walltime_s=res.log.walltime_s, descriptors={}, per_cell_fitness={},
                    generation=g, island=island, stage_reached="validate",
                    status="invalid-schedule")
                continue
            a = self.cascade.stage_a(child, self.stage_a_cells, self.cfg.stage_a_seeds)
            if not a.passed:
                self.archive.add(candidate_id=h, source=child, descriptors=desc,
                                 fitness=float("-inf"), worst_cell=float("-inf"),
                                 feasible=False, generation=g, island=island,
                                 parent_ids=[parent.candidate_id], engine=res.log.engine)
                self._log(h, parent, res, desc, {}, g, island, "A", "fast-rejected")
                continue
            b = self.cascade.stage_b(child, self.stage_b_cells, self.cfg.stage_b_seeds)
            per_cell = _fw_percell(b.per_cell)
            accepted, coord = self.archive.add(
                candidate_id=h, source=child, descriptors=desc, fitness=b.fitness,
                worst_cell=b.worst_cell, feasible=b.passed, generation=g,
                island=island, parent_ids=[parent.candidate_id], engine=res.log.engine,
                per_cell=per_cell)
            status = "elite" if (b.passed and accepted) else (
                "feasible" if b.passed else "infeasible")
            # Stage C only for a newly-accepted feasible elite
            if b.passed and accepted:
                # holdout floor: ROWP measured floor (real) / cheap-cell ~0.1 (dry)
                c = self.cascade.stage_c(child, self.holdout_cell,
                                         self.cfg.stage_c_seeds,
                                         self.cfg.holdout_floor_gwh)
                status = "elite+stageC"
                per_cell = dict(per_cell, _stageC=_fw_stagec(c))
            self._log(h, parent, res, desc, per_cell, g, island, "C" if status.endswith("C") else "B", status)

        self.generation += 1
        if self.generation >= self.cfg.generations:
            self.done = True
        self.checkpoint()
        return "DONE" if self.done else "GEN_COMPLETE"

    def _log(self, h, parent, res, desc, per_cell, g, island, stage, status):
        self.lineage.log_candidate(
            candidate_id=h, parent_ids=[parent.candidate_id], engine=res.log.engine,
            model=res.log.model, prompt_tokens=res.log.prompt_tokens,
            completion_tokens=res.log.completion_tokens, usd=res.log.usd,
            walltime_s=res.log.walltime_s, descriptors=desc, per_cell_fitness=per_cell,
            generation=g, island=island, stage_reached=stage, status=status)

    # ── run driver (one generation per call by default) ───────────────
    def run(self, max_generations: Optional[int] = 1) -> str:
        if not self.seeded:
            self.seed_gen0()
            self.checkpoint()
            if max_generations == 0:
                return "SEEDED"     # seed-only chunk (splits gen-0 seeding off)
        elif max_generations == 0:
            return "SEEDED"
        n = 0
        while not self.done and not self.aborted:
            status = self.step_generation()
            if status == "ABORT":
                return "ABORT"
            n += 1
            if max_generations is not None and n >= max_generations:
                return "PAUSED" if not self.done else "DONE"
        return "ABORT" if self.aborted else "DONE"


# ── helpers ───────────────────────────────────────────────────────────
def _import_fn(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def _fn_from_source(src):
    import importlib.util, tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        return _import_fn(path)
    finally:
        os.unlink(path)


def _fw_percell(per_cell):
    """Firewall-safe per-cell summary for the lineage/prompt: %-scores +
    feasibility, NO holdout/test AEP (training-cell means are allowed)."""
    out = {}
    for cell, sc in per_cell.items():
        out[cell] = {"score_pct": (round(sc["score"], 4) if sc["score"] != float("-inf") else None),
                     "feasible": sc.get("feasible")}
    return out


def _fw_stagec(c):
    """Strip the _firewalled block: only booleans + cell id cross the firewall."""
    return {k: v for k, v in c.items() if k != "_firewalled"}
