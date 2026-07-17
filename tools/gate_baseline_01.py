"""Gate the 500-start baseline seeds at a clean 0.1 m tolerance and write the
best-feasible AEP per cell — replacing the soft penalty-budget criterion
(boundary_penalty < 1e-3) with a per-turbine max-outside gate that matches
fig_aep_dominance.

Reads per-seed .out files (with max_out_m/min_dist_m) from a directory, gates
at TOL, takes the best AEP. Output: results/baselines_01tol.json.

Usage: pixi run python tools/gate_baseline_01.py <seed-dir>
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 0.1
SPACING = {"dei_n50": 960.0, "rowp_n74": 792.0}

seed_dir = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(ROOT, "results/baseline_01tol")
out = {}
for key, minsp in SPACING.items():
    files = glob.glob(os.path.join(seed_dir, f"{key}_seed*.out"))
    seeds = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "aep" not in d or "max_out_m" not in d:
            continue
        seeds.append(d)
    if not seeds:
        print(f"{key}: no seed data in {seed_dir}")
        continue
    feas = [s for s in seeds if s["max_out_m"] <= TOL and s["min_dist_m"] >= minsp - TOL]
    best = max((s["aep"] for s in feas), default=None)
    # also strict (SDF<=0) and the current soft criterion for context
    strict = [s for s in seeds if s["max_out_m"] <= 0.0 and s["min_dist_m"] >= minsp]
    best_strict = max((s["aep"] for s in strict), default=None)
    out[key] = {
        "n_starts": len(seeds), "tol_m": TOL,
        "best_aep_01tol": best, "n_feasible_01tol": len(feas),
        "best_aep_strict": best_strict, "n_feasible_strict": len(strict),
    }
    print(f"{key}: best@0.1m={best} ({len(feas)}/{len(seeds)} feas) | "
          f"strict={best_strict} ({len(strict)}/{len(seeds)})")

json.dump(out, open(os.path.join(ROOT, "results/baselines_01tol.json"), "w"), indent=1)
print("wrote results/baselines_01tol.json")
