"""ParqueFicticio schedule matrix: 3 schedules × 6 turbine counts × 4 roses.

Schedules:
  baseline — results/seed_schedule.py (TopFarm-style coupled decay)
  gemini   — runs/gemini_cli_5hr/iter_192.py (deployed)
  claude   — runs/schedule_only_5hr/iter_192.py (deployed dual-bump)

Counts: 10, 15, 20, 25, 30, 35.
Roses: the paper's four (dei, rowp, omnidir, uniform), taken verbatim from
the matrix problem files and applied to the ParqueFicticio site (V80,
five disconnected inclusion zones, 2 D spacing).

Scoring: deterministic full-rose AEP (GWh) + multi-zone feasibility
(all turbines inside a zone, min spacing respected).

Resume-safe: writes parqo/parqo_matrix.json incrementally.

Usage:
    PYTHONPATH=dependencies/pixwake/src pixi run python parqo/run_parqo_matrix.py
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "dependencies/pixwake/src"))
sys.path.insert(0, os.path.join(ROOT, "playground"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from harness import build_sim                # noqa: E402
from skeleton_multizone import (             # noqa: E402
    run_with_schedule, multizone_sdf,
)

SCHEDULES = {
    "baseline": os.path.join(ROOT, "results/seed_schedule.py"),
    "gemini":   os.path.join(ROOT, "runs/gemini_cli_5hr/iter_192.py"),
    "claude":   os.path.join(ROOT, "runs/schedule_only_5hr/iter_192.py"),
}
COUNTS = [10, 15, 20, 25, 30, 35]
ROSES = {
    "dei":     "results/matrix/problem_dei_n50_rosedei.json",
    "rowp":    "results/matrix/problem_dei_n50_roserowp.json",
    "omnidir": "results/matrix/problem_dei_n50_roseomnidir.json",
    "uniform": "results/matrix/problem_dei_n50_roseuniform.json",
}
OUT = os.path.join(HERE, "parqo_matrix.json")


def load_schedule_fn(path):
    spec = importlib.util.spec_from_file_location("sched_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def feasibility(x, y, zones, min_spacing):
    x = np.asarray(x); y = np.asarray(y)
    zones_j = [jnp.asarray(z) for z in zones]
    sdf = np.asarray(multizone_sdf(jnp.array(x), jnp.array(y), zones_j))
    inside = sdf <= 1e-6  # inside any zone (sdf <= 0 inside)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2) + np.eye(len(x)) * 1e9
    min_d = float(dist.min())
    return bool(inside.all() and min_d >= min_spacing - 1e-6), float(np.max(sdf)), min_d


def main():
    problem = json.load(open(os.path.join(HERE, "problem_parqo.json")))
    zones = problem["inclusion_zones"]
    min_spacing = float(problem["min_spacing_m"])
    sim = build_sim(problem)

    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    fns = {k: load_schedule_fn(p) for k, p in SCHEDULES.items()}
    roses = {k: json.load(open(os.path.join(ROOT, p)))["wind_rose"]
             for k, p in ROSES.items()}

    total = len(SCHEDULES) * len(COUNTS) * len(ROSES)
    done = 0
    t0 = time.time()
    for rose_name, wr in roses.items():
        wd = jnp.array(wr["directions_deg"], dtype=jnp.float64)
        ws = jnp.array(wr["speeds_ms"], dtype=jnp.float64)
        wts = jnp.array(wr["weights"], dtype=jnp.float64)
        wts = wts / jnp.sum(wts)

        def aep_gwh(x, y):
            r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
            p = r.power()[:, :len(x)]
            return float(jnp.sum(jnp.sum(p, axis=1) * wts) * 8760 / 1e6)  # kW -> GWh

        for n in COUNTS:
            for sched_name, fn in fns.items():
                done += 1
                key = f"{sched_name}|{rose_name}|n{n}"
                if key in results:
                    continue
                t1 = time.time()
                try:
                    x, y = run_with_schedule(fn, sim, n, zones, min_spacing,
                                             wd, ws, wts, seed=0)
                    feas, max_sdf, min_d = feasibility(x, y, zones, min_spacing)
                    aep = aep_gwh(jnp.array(np.asarray(x)), jnp.array(np.asarray(y)))
                    results[key] = {
                        "schedule": sched_name, "rose": rose_name, "n": n,
                        "aep_gwh": round(aep, 4), "feasible": feas,
                        "max_sdf_m": round(max_sdf, 3),
                        "min_dist_m": round(min_d, 2),
                        "x": [float(v) for v in np.asarray(x)],
                        "y": [float(v) for v in np.asarray(y)],
                        "time_s": round(time.time() - t1, 1),
                    }
                except Exception as e:
                    results[key] = {"schedule": sched_name, "rose": rose_name,
                                    "n": n, "error": str(e)[:200]}
                with open(OUT, "w") as f:
                    json.dump(results, f)
                r = results[key]
                tag = (f"AEP={r['aep_gwh']:.2f} {'FEAS' if r['feasible'] else 'INFEAS'} "
                       f"min_d={r['min_dist_m']}m {r['time_s']}s"
                       if "aep_gwh" in r else f"ERR {r['error'][:60]}")
                print(f"[{done}/{total}] {key}: {tag}", flush=True)

    print(f"\nWall: {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
