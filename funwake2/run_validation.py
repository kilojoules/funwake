"""Post-hoc validation of a discovered schedule vs the native c*D baseline, paired
by seed, across training cells + the held-out ROWP. This is the deployment-decision
evaluation (I run it; nothing here is fed to a mutator). Per cell: candidate mean
AEP vs native mean AEP, paired Δ%, and candidate feasibility. Native baseline comes
from baselines_g2.json where available (paired, same seeds), else run fresh.

Resumable + chunked (per-cell results cached) so a kill doesn't lose progress.

  pixi run python funwake2/run_validation.py --cand <sched.py> --cells dei_n50 --seeds 0..9
"""
import argparse
import json
import os
import statistics
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluator as E  # noqa: E402

_BASE = json.load(open(os.path.join(_THIS, "controller", "baselines_g2.json")))["cells"]


def eval_cell(cand_fn, native_fn, cell, seeds, steps):
    rows = []
    bg = _BASE.get(cell, {}).get("seeds", {})
    for s in seeds:
        c = E.evaluate(cell, cand_fn, seed=s, total_steps=steps, gamma_min=0.01)
        if str(s) in bg:                       # paired native from the frozen baseline
            nb, nfeas = bg[str(s)], True
        else:                                  # held-out cell: run native fresh
            nr = E.evaluate(cell, native_fn, seed=s, total_steps=steps, gamma_min=0.01)
            nb, nfeas = nr["aep_gwh"], nr["feasible"]
        rows.append({"seed": s, "cand_aep": c["aep_gwh"], "cand_feas": c["feasible"],
                     "native_aep": nb, "native_feas": nfeas,
                     "delta": round(c["aep_gwh"] - nb, 4)})
    cm = statistics.fmean(r["cand_aep"] for r in rows)
    nm = statistics.fmean(r["native_aep"] for r in rows)
    return {"cell": cell, "n_seeds": len(seeds), "cand_mean": round(cm, 4),
            "native_mean": round(nm, 4), "delta_pct": round(100.0 * (cm - nm) / nm, 4),
            "cand_feasible": f"{sum(r['cand_feas'] for r in rows)}/{len(seeds)}",
            "seeds": seeds, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(_THIS, "state", "validation", "iter04.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"cand": a.cand, "cells": {}}
    cand_fn = E.load_schedule(a.cand)
    native_fn = E.load_schedule(os.path.join(_THIS, "seeds", "native.py"))
    for cell in a.cells:
        key = f"{cell}@{len(a.seeds)}"
        if key in res["cells"]:
            print(f"  {cell}: (cached)", flush=True); continue
        r = eval_cell(cand_fn, native_fn, cell, a.seeds, a.steps)
        res["cells"][key] = r
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"  {cell:22s} cand {r['cand_mean']} vs native {r['native_mean']} = "
              f"{r['delta_pct']:+.4f}%  feasible {r['cand_feasible']}", flush=True)
    print("\n=== VALIDATION SO FAR ===", flush=True)
    for key, r in res["cells"].items():
        farm = "HELD-OUT" if key.startswith("rowp") else "train"
        print(f"  [{farm:8s}] {key:26s} {r['delta_pct']:+.4f}%  feas {r['cand_feasible']}")


if __name__ == "__main__":
    main()
