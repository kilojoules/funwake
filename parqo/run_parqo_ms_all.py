"""ParqueFicticio multistart baseline: seed schedule × 50 init seeds per cell.

Mirrors the paper's baseline construction (multi-start SGD, best feasible
kept) for the parqo matrix cells: 6 counts × 4 roses, K=50 starts each.
Feasibility: practical tolerance (zone violation <= 0.1 m, spacing deficit
<= 0.1 m) — same convention as the schedule-matrix analysis.

Writes parqo/parqo_baseline_ms.json (resume-safe per cell).
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "playground/pixwake/src"))
sys.path.insert(0, os.path.join(ROOT, "playground"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from harness import build_sim                # noqa: E402
from skeleton_multizone import run_with_schedule, multizone_sdf  # noqa: E402

K = 50
TOL = 0.1
COUNTS = [10, 15, 20, 25, 30, 35]
ROSES = {
    "dei":     "results/matrix/problem_dei_n50_rosedei.json",
    "rowp":    "results/matrix/problem_dei_n50_roserowp.json",
    "omnidir": "results/matrix/problem_dei_n50_roseomnidir.json",
    "uniform": "results/matrix/problem_dei_n50_roseuniform.json",
}
import sys as _sys
SCHED_NAME = _sys.argv[1] if len(_sys.argv) > 1 else "baseline"
SCHED_PATH = {
    "baseline": os.path.join(ROOT, "results/seed_schedule.py"),
    "claude":   os.path.join(ROOT, "runs/schedule_only_5hr/iter_192.py"),
    "gemini":   os.path.join(ROOT, "runs/gemini_cli_5hr/iter_118.py"),
}[SCHED_NAME]
OUT = os.path.join(HERE, f"parqo_ms_{SCHED_NAME}.json")

spec = importlib.util.spec_from_file_location("sched", SCHED_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
schedule_fn = mod.schedule_fn


def main():
    problem = json.load(open(os.path.join(HERE, "problem_parqo.json")))
    zones = problem["inclusion_zones"]
    zones_j = [jnp.asarray(z) for z in zones]
    min_spacing = float(problem["min_spacing_m"])
    sim = build_sim(problem)

    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    for rose_name, rel in ROSES.items():
        wr = json.load(open(os.path.join(ROOT, rel)))["wind_rose"]
        wd = jnp.array(wr["directions_deg"], dtype=jnp.float64)
        ws = jnp.array(wr["speeds_ms"], dtype=jnp.float64)
        wts = jnp.array(wr["weights"], dtype=jnp.float64)
        wts = wts / jnp.sum(wts)

        def aep_gwh(x, y):
            r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
            p = r.power()[:, :len(x)]
            return float(jnp.sum(jnp.sum(p, axis=1) * wts) * 8760 / 1e6)

        for n in COUNTS:
            key = f"{rose_name}|n{n}"
            if key in results:
                continue
            t0 = time.time()
            seeds_out = []
            for seed in range(K):
                try:
                    x, y = run_with_schedule(schedule_fn, sim, n, zones,
                                             min_spacing, wd, ws, wts,
                                             seed=seed)
                except Exception:
                    continue
                xa = np.asarray(x); ya = np.asarray(y)
                sdf = float(np.max(np.asarray(
                    multizone_sdf(jnp.array(xa), jnp.array(ya), zones_j))))
                dx = xa[:, None] - xa[None, :]; dy = ya[:, None] - ya[None, :]
                md = float((np.sqrt(dx**2 + dy**2) + np.eye(n) * 1e9).min())
                aep = aep_gwh(jnp.array(xa), jnp.array(ya))
                seeds_out.append({"seed": seed, "aep_gwh": round(aep, 4),
                                  "max_sdf_m": round(sdf, 3),
                                  "min_dist_m": round(md, 2)})
            n_feas = sum(1 for r in seeds_out
                         if r["max_sdf_m"] <= TOL
                         and r["min_dist_m"] >= min_spacing - TOL)
            feas = [r for r in seeds_out if r["max_sdf_m"] <= TOL
                    and r["min_dist_m"] >= min_spacing - TOL]
            best = max(feas, key=lambda r: r["aep_gwh"]) if feas else None
            results[key] = {
                "rose": rose_name, "n": n, "K": K, "tol_m": TOL,
                "n_feasible_starts": n_feas,
                "best": best,
                "seeds": seeds_out,
                "time_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(results, f)
            tag = (f"best={best['aep_gwh']:.2f} (seed {best['seed']}, "
                   f"{n_feas}/{K} feas)" if best else f"NO FEASIBLE ({n_feas}/{K})")
            print(f"{key}: {tag} [{results[key]['time_s']}s]", flush=True)

    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
