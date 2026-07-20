"""Matrix multistart: schedule × K init seeds through the funwake skeleton,
best feasible AEP per cell. Makes the schedule-vs-baseline comparison FAIR
(same K starts, same skeleton) instead of best-of-500 baseline vs 1-run
schedule.

One LSF array task = one matrix cell (0..47). Per cell, runs all three
schedules (baseline seed / claude / gemini) × K starts, keeps best feasible.
Deterministic full-rose AEP + boundary/spacing feasibility (practical tol).

Usage:
    PYTHONPATH=dependencies/pixwake/src:playground pixi run python \\
        tools/run_matrix_ms.py --cell-idx <0..47> --K 50 \\
        --out results/matrix/ms/<key>.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dependencies/pixwake/src"))
sys.path.insert(0, os.path.join(ROOT, "playground"))

CELLS = [("rowp", n, r) for r in ["uniform", "dei"] for n in [30,40,50,60,70,80]]  # 12 Weibull pilot

SCHEDULES = {
    "baseline": "results/seed_schedule.py",
    "claude":   "runs/schedule_only_5hr/iter_192.py",
    "gemini":   "runs/gemini_cli_5hr/iter_118.py",
}
TOL = 0.1  # practical feasibility tolerance (m)


def load_fn(rel):
    spec = importlib.util.spec_from_file_location("s", os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.schedule_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-idx", type=int, required=True)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="run ONLY this schedule")
    ap.add_argument("--redo", default="", help="comma-sep schedules to recompute")
    args = ap.parse_args()
    global SCHEDULES
    if args.only:
        SCHEDULES = {args.only: SCHEDULES[args.only]}

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from harness import build_sim
    from skeleton import run_with_schedule
    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from pixwake.optim.boundary import polygon_sdf

    farm, n, rose = CELLS[args.cell_idx]
    key = f"{farm}_n{n}_rose{rose}"
    problem = json.load(open(os.path.join(ROOT, f"results/matrix/problem_{key}_weibull.json")))
    sim = build_sim(problem)
    bnd = jnp.array(problem["boundary_vertices"], dtype=jnp.float64)
    ms = float(problem["min_spacing_m"])
    wr = problem["wind_rose"]
    wd = jnp.array(wr["directions_deg"]); ws = jnp.array(wr["speeds_ms"])
    wts = jnp.array(wr["weights"]); wts = wts / jnp.sum(wts)

    def det_aep(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return float(jnp.sum(jnp.sum(p, axis=1) * wts) * 8760 / 1e6)

    def feas(x, y):
        xa = np.asarray(x); ya = np.asarray(y)
        sdf = np.asarray(polygon_sdf(jnp.array(xa), jnp.array(ya), bnd))
        max_out = float(np.max(sdf))            # >0 outside polygon
        dx = xa[:, None]-xa[None, :]; dy = ya[:, None]-ya[None, :]
        md = float((np.sqrt(dx**2+dy**2)+np.eye(len(xa))*1e9).min())
        return (max_out <= TOL and md >= ms - TOL), round(max_out, 3), round(md, 2)

    results = {"cell": key, "K": args.K, "tol_m": TOL, "schedules": {}}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        for _r in [r for r in args.redo.split(",") if r]:
            results["schedules"].pop(_r, None)
    t0 = time.time()
    for sname, rel in SCHEDULES.items():
        if sname in results["schedules"]:
            continue
        fn = load_fn(rel)
        seeds_out = []
        for seed in range(args.K):
            try:
                x, y = run_with_schedule(fn, sim, n, bnd, ms, wd, ws, wts, seed=seed)
            except Exception as e:
                seeds_out.append({"seed": seed, "error": str(e)[:80]}); continue
            ok, mo, md = feas(x, y)
            seeds_out.append({"seed": seed, "aep_gwh": round(det_aep(x, y), 3),
                              "feasible": ok, "max_out_m": mo, "min_dist_m": md})
        good = [r for r in seeds_out if r.get("feasible")]
        best = max(good, key=lambda r: r["aep_gwh"]) if good else None
        results["schedules"][sname] = {
            "best": best, "n_feasible": len(good),
            "seeds": seeds_out,
            "best_infeas_aep": (max((r["aep_gwh"] for r in seeds_out if "aep_gwh" in r),
                                    default=None) if not good else None),
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        tag = f"best={best['aep_gwh']:.1f} ({len(good)}/{args.K} feas)" if best else f"NO FEAS ({len(good)}/{args.K})"
        print(f"{key} {sname}: {tag} [{time.time()-t0:.0f}s]", flush=True)
    print(f"cell {key} done", flush=True)


if __name__ == "__main__":
    main()
