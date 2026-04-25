"""Aggregate per-seed matrix results.

Merges shard JSONs (if present), computes mean/std/feasible_fraction
per (script, cell), and isolates the uniform-wind cells for the
systematic-vs-noise check.

Usage:
    pixi run python experiments/D_matrix_multi_seed/aggregate.py
"""
import glob
import json
import os
import statistics
import collections


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all():
    merged = {}
    for path in glob.glob(os.path.join(EXP_DIR, "results*.json")):
        if path.endswith("summary.json"):
            continue
        d = json.load(open(path))
        merged.update(d)
    return merged


def main():
    raw = load_all()
    if not raw:
        print("[D] No results.json or shard files found. Run eval.py first.")
        return

    # group by (label, farm, n, rose)
    groups = collections.defaultdict(list)
    for rec in raw.values():
        if "aep_gwh" not in rec:
            continue
        key = (rec["label"], rec["farm"], rec["n"], rec["rose"])
        groups[key].append(rec)

    cells = []
    uniform_cells = []
    for (label, farm, n, rose), recs in sorted(groups.items()):
        aeps = [r["aep_gwh"] for r in recs]
        feas = [int(bool(r.get("feasible"))) for r in recs]
        row = {
            "label": label, "farm": farm, "n": n, "rose": rose,
            "n_seeds": len(recs),
            "aep_mean": round(statistics.mean(aeps), 3),
            "aep_std": round(statistics.pstdev(aeps), 3) if len(aeps) > 1 else 0.0,
            "aep_min": round(min(aeps), 3),
            "aep_max": round(max(aeps), 3),
            "feasible_fraction": round(sum(feas) / len(feas), 3),
        }
        cells.append(row)
        if rose == "uniform":
            uniform_cells.append(row)

    summary = {
        "n_cells": len(cells),
        "cells": cells,
        "uniform_wind_analysis": {
            "n_cells": len(uniform_cells),
            "rows": uniform_cells,
            "all_systematic": all(
                r["feasible_fraction"] in (0.0, 1.0) for r in uniform_cells
            ),
        },
    }
    out = os.path.join(EXP_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D] Wrote {out}")
    print(f"[D] Uniform-wind systematic? {summary['uniform_wind_analysis']['all_systematic']}")
    for r in uniform_cells:
        print(f"    {r['label']:30s} {r['farm']:5s} N={r['n']:<4d} feas={r['feasible_fraction']:.2f}  aep={r['aep_mean']:.1f}±{r['aep_std']:.1f}")


if __name__ == "__main__":
    main()
