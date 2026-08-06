"""Local FRESH-PROCESS validation — for schedules that leak a cached tracer under
in-process reuse (e.g. it190's _decay_table). Each candidate eval runs in a brand-new
interpreter (maxtasksperchild=1), so no cross-eval tracer leak. Compares to native:
baselines_g2 for training cells (same reference the specialists used), fresh native
for held-out ROWP + any missing seeds. Writes the run_validation.py output format so
select_deploy.py picks it up directly.

  pixi run python funwake2/validate_freshproc.py --cand seed_port_gbar_iter190.py \
      --out funwake2/state/validation/port190.json --jobs 4
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from funwake2.gbar_eval import _eval_one            # (cell,seed,sched,steps)->dict
_BASE = json.load(open(os.path.join(_THIS, "controller", "baselines_g2.json")))["cells"]
_NATIVE = os.path.join(_THIS, "seeds", "native.py")


def _eval(sched, cell, seeds, steps, jobs):
    tasks = [(cell, s, os.path.abspath(sched), steps) for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(jobs, len(tasks)), maxtasksperchild=1) as pool:
        res = pool.map(_eval_one, tasks)
    return {r["seed"]: r for r in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells10", nargs="+", default=["dei_n50", "rowp_n74"])
    ap.add_argument("--cells5", nargs="+",
                    default=["parque_n20", "parque_n10_omnidir", "dei_n80_omnidir", "dei_n50_uniform"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"cand": a.cand, "cells": {}}
    plan = [(c, list(range(10))) for c in a.cells10] + [(c, list(range(5))) for c in a.cells5]
    for cell, seeds in plan:
        key = f"{cell}@{len(seeds)}"
        if key in res["cells"]:
            print(f"  {key}: cached", flush=True); continue
        cand_ev = _eval(a.cand, cell, seeds, a.steps, a.jobs)
        # native: baselines_g2 where available, else fresh
        nat_ev = {}
        if cell in _BASE:
            for s in seeds:
                if str(s) in _BASE[cell]["seeds"]:
                    nat_ev[s] = {"aep": _BASE[cell]["seeds"][str(s)], "feasible": True}
        miss = [s for s in seeds if s not in nat_ev]
        if miss:
            nat_ev.update(_eval(_NATIVE, cell, miss, a.steps, a.jobs))
        rows = [{"seed": s, "cand_aep": cand_ev[s]["aep"], "cand_feas": cand_ev[s]["feasible"],
                 "native_aep": nat_ev[s]["aep"], "native_feas": nat_ev[s].get("feasible", True)}
                for s in seeds]
        cm = statistics.fmean(r["cand_aep"] for r in rows)
        nm = statistics.fmean(r["native_aep"] for r in rows)
        res["cells"][key] = {"cell": cell, "n_seeds": len(seeds), "cand_mean": round(cm, 4),
                             "native_mean": round(nm, 4), "delta_pct": round(100 * (cm - nm) / nm, 4),
                             "cand_feasible": f"{sum(r['cand_feas'] for r in rows)}/{len(seeds)}",
                             "rows": rows}
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"  {key:22s} {res['cells'][key]['delta_pct']:+.4f}%  feas {res['cells'][key]['cand_feasible']}",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
