#!/usr/bin/env python
"""Dry-run driver — machinery validation with a MOCK mutator (NO LLM spend).

Two modes:
  * default (real jax evals on the smoke cells dei_n50 + parque_n20) — end-to-end
    integration incl. content-addressed cache + resume.
  * --fake-eval — a deterministic, jax-free stand-in evaluator so the controller
    state machine (cascade order, archive binning, lineage, cost abort, resume
    bit-identity) is validated instantly and chunk-free.

One generation per invocation by default (checkpoint + exit) so every call stays
well under the watchdog; re-invoke with the same --state-dir to resume. --run-all
runs to completion (use only with --fake-eval or tiny real settings).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_FW2 = os.path.dirname(_THIS)
_ROOT = os.path.dirname(_FW2)
for _p in (_ROOT, _FW2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from funwake2.controller import config as C          # noqa: E402
from funwake2.controller.cascade import Cascade       # noqa: E402
from funwake2.controller.controller import Controller  # noqa: E402
from funwake2.controller.engines.mock import MockEngine  # noqa: E402

# deterministic, jax-free pseudo-AEP per cell (scale constants)
_FAKE_BASE = {"dei_n50": 5560.0, "parque_n20": 231.0, "parque_n10_omnidir": 118.0,
              "dei_n80_omnidir": 8600.0, "rowp_n74": 4263.0}


def make_fake_eval(infeasible_cells=None, infeasible_ref=None):
    """Return a deterministic evaluate(cell, schedule_fn, seed, steps, gamma)
    that never imports jax. AEP is a stable function of the schedule's bytecode
    hash + cell + seed. `infeasible_ref` (a cell) makes the NATIVE reference
    infeasible there — to exercise the scale-constant fitness patch."""
    infeasible_cells = set(infeasible_cells or [])

    def fake(cell, schedule_fn, seed=0, total_steps=8000, gamma_min=0.01):
        code = getattr(schedule_fn, "__code__", None)
        raw = (code.co_code if code else b"") + cell.encode() + str(seed).encode()
        hi = int(hashlib.sha256(raw).hexdigest()[:12], 16)
        base = _FAKE_BASE.get(cell, 1000.0)
        # small NON-NEGATIVE delta (0..+0.1%): stays within the stage-A noise
        # floor so feasible candidates aren't spuriously fast-rejected, while
        # still giving the archive fitness structure. Deterministic (hash-based).
        aep = round(base * (1.0 + (hi % 1000) / 1e6), 4)
        is_ref = "_baseline" in getattr(schedule_fn, "__name__", "")  # not reliable
        feasible = cell not in infeasible_cells
        # reference-infeasible scenario keyed by a marker on the cell
        if infeasible_ref and cell == infeasible_ref and _looks_like_ref(schedule_fn):
            feasible = False
        return {"cell": cell, "seed": seed, "steps": total_steps,
                "gamma_min": gamma_min, "aep_gwh": aep, "feasible": feasible,
                "D": 240.0, "n": 50}

    return fake


def _looks_like_ref(fn):
    # the native reference seed module defines schedule_fn with lr0 = _C * D
    try:
        return "native" in (fn.__code__.co_filename or "").lower()
    except Exception:
        return False


def build_config(args) -> C.RunConfig:
    return C.RunConfig(
        run_id=args.run_id,
        state_dir=args.state_dir,
        cache_dir=args.cache_dir,
        lineage_path=args.lineage,
        dry_run=True,
        dry_cells=args.cells,
        generations=args.gens,
        proposals_per_gen=args.proposals,
        stage_a_seeds=args.a_seeds,
        stage_b_seeds=args.b_seeds,
        stage_c_seeds=args.c_seeds,
        num_islands=args.islands,
        max_usd=args.max_usd,
        max_tokens=args.max_tokens,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="dry_run")
    p.add_argument("--state-dir", default="funwake2/state/dry/state")
    p.add_argument("--cache-dir", default="funwake2/state/dry/cache")
    p.add_argument("--lineage", default="funwake2/state/dry/lineage.jsonl")
    p.add_argument("--cells", nargs="+", default=["dei_n50", "parque_n20"])
    p.add_argument("--gens", type=int, default=2)
    p.add_argument("--proposals", type=int, default=2)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--a-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--b-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--c-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-usd", type=float, default=1000.0)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--seed-only", action="store_true",
                   help="seed generation 0 then checkpoint+exit (chunking)")
    p.add_argument("--fake-eval", action="store_true")
    p.add_argument("--fake-infeasible-ref", default=None,
                   help="cell whose native reference is infeasible (fitness patch test)")
    args = p.parse_args()

    cfg = build_config(args)
    cache = None
    if args.fake_eval:
        cascade = Cascade(cfg, evaluate_fn=make_fake_eval(
            infeasible_ref=args.fake_infeasible_ref))
    else:
        cascade = Cascade(cfg)
    ctrl = Controller(cfg, MockEngine(), cascade)
    if args.seed_only:
        mg = 0
    elif args.run_all:
        mg = None
    else:
        mg = 1
    status = ctrl.run(max_generations=mg)

    print(f"STATUS={status} generation={ctrl.generation} done={ctrl.done} "
          f"aborted={ctrl.aborted}")
    print(f"COST usd={ctrl.cost.state.usd:.4f}/{cfg.max_usd} "
          f"tokens={ctrl.cost.state.tokens}/{cfg.max_tokens} "
          f"calls={ctrl.cost.state.n_calls} abort={ctrl.cost.should_abort()}")
    print(f"CACHE hits={ctrl.cache.hits} misses={ctrl.cache.misses}")
    print("ARCHIVE " + json.dumps(ctrl.archive.summary()))
    best = ctrl.archive.best()
    if best:
        print(f"BEST fitness={best.fitness:.4f} worst_cell={best.worst_cell:.4f} "
              f"cell={ctrl.archive._coord_key(best.coord)}")


if __name__ == "__main__":
    main()
