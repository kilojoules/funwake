#!/usr/bin/env python
"""Aggregate the post-G8 native baseline (funwake2/state/g2_baseline/*.json) into
per-cell mean/std/feas and write the FROZEN in-search baseline table
funwake2/controller/baselines_g2.json (per-seed AEP retained for paired scoring).

Usage: pixi run python funwake2/aggregate_g2.py            # aggregate + write
       pixi run python funwake2/aggregate_g2.py --progress # just show counts
"""
import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
INDIR = os.path.join(HERE, "state", "g2_baseline")
OUT = os.path.join(HERE, "controller", "baselines_g2.json")

FROZEN_CELLS = ["dei_n50", "dei_n80_omnidir", "dei_n120_rosedei", "dei_n50_uniform",
                "parque_n20", "parque_n14_uniform", "parque_n10_omnidir"]


def collect():
    per_cell = {}
    for path in sorted(glob.glob(os.path.join(INDIR, "native_*_g0.01_s8000_seed*.json"))):
        rec = json.load(open(path))
        if "error" in rec:
            per_cell.setdefault(rec["cell"], {}).setdefault("errors", []).append(rec["seed"])
            continue
        c = per_cell.setdefault(rec["cell"], {"seeds": {}})
        c["seeds"][str(rec["seed"])] = rec["aep_gwh"]
        c.setdefault("feas", {})[str(rec["seed"])] = bool(rec["feasible"])
        c.setdefault("_meta", {"D": rec.get("D"), "n": rec.get("n"),
                               "min_spacing": rec.get("min_spacing")})
    return per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    per_cell = collect()

    print(f"{'cell':22s} {'n':>4s} {'seeds':>6s} {'mean':>10s} {'std':>7s} "
          f"{'sem':>6s} {'feas':>6s}")
    out = {"gamma_min": 0.01, "total_steps": 8000, "source": "state/g2_baseline",
           "note": "post-G8 (float32 alpha0 canonicalization) in-search c*D baseline",
           "cells": {}}
    for cell in FROZEN_CELLS:
        c = per_cell.get(cell)
        if not c or "seeds" not in c:
            print(f"{cell:22s} {'--':>4s}  (no results yet)")
            continue
        seeds = c["seeds"]
        aeps = [seeds[k] for k in sorted(seeds, key=int)]
        feas = c.get("feas", {})
        nfeas = sum(1 for v in feas.values() if v)
        mean = statistics.fmean(aeps)
        std = statistics.pstdev(aeps) if len(aeps) > 1 else 0.0
        sem = std / (len(aeps) ** 0.5) if aeps else 0.0
        meta = c.get("_meta", {})
        print(f"{cell:22s} {str(meta.get('n')):>4s} {len(aeps):>6d} "
              f"{mean:>10.3f} {std:>7.3f} {sem:>6.3f} {nfeas:>3d}/{len(aeps)}")
        out["cells"][cell] = {
            "n": meta.get("n"), "D": meta.get("D"),
            "min_spacing": meta.get("min_spacing"),
            "mean": round(mean, 4), "std": round(std, 4), "sem": round(sem, 4),
            "feas": f"{nfeas}/{len(aeps)}", "n_seeds": len(aeps),
            "seeds": {k: seeds[k] for k in sorted(seeds, key=int)},
        }
    if not args.progress:
        complete = all(cell in out["cells"] and out["cells"][cell]["n_seeds"] >= 10
                       for cell in FROZEN_CELLS)
        out["complete_10seed_all_cells"] = complete
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {OUT}  (complete={complete})")


if __name__ == "__main__":
    main()
