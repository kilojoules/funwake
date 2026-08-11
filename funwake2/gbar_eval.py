"""Parallel portfolio evaluator (gbar). Evaluates a schedule on (cells x seeds)
CONCURRENTLY across cores (ProcessPoolExecutor, spawn — each worker gets its own
JAX context, avoiding fork deadlocks) and writes raw per-(cell,seed) AEP +
feasibility JSON. This is where the speedup comes from: the 5x3=15 evals that run
sequentially on the Mac (~400s) run in parallel here (~one eval's wall-time).

Used for BOTH candidate scoring and native-baseline computation. The orchestrator
computes farm-balanced score_c from these vs GBAR-NATIVE baselines (same-platform,
so cross-platform AEP drift never enters the metric).

  python funwake2/gbar_eval.py --schedule sched.py --cells dei_n50 parque_n20 \
      --seeds 0 1 2 --steps 8000 --out scores.json --jobs 16
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _eval_one(task):
    # pin each worker to a single thread BEFORE jax imports, so N parallel workers
    # cleanly use N cores instead of each grabbing all cores (XLA CPU default) and
    # thrashing. Set at process start (spawn) — effective for this worker's jax.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    cell, seed, sched_path, steps = task
    import evaluator as E                       # imported inside worker (spawn)
    try:                                        # register matrix farms if present
        import matrix_cells
        matrix_cells.register(E)
    except Exception:
        pass
    fn = E.load_schedule(sched_path)
    r = E.evaluate(cell, fn, seed=seed, total_steps=steps, gamma_min=0.01)
    return {"cell": cell, "seed": int(seed), "aep": float(r["aep_gwh"]),
            "feasible": bool(r["feasible"]),
            "min_dist_m": float(r.get("min_dist_m", 0.0) or 0.0),
            "boundary_penalty": float(r.get("boundary_penalty", 0.0) or 0.0),
            "min_spacing": float(r.get("min_spacing", 0.0) or 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()

    sched = os.path.abspath(a.schedule)
    tasks = [(c, s, sched, a.steps) for c in a.cells for s in a.seeds]
    t0 = time.time()
    # maxtasksperchild=1: each eval runs in a brand-new interpreter, so a schedule
    # that caches a traced value at module scope (e.g. it190's _decay_table) can
    # never leak that tracer into the next eval. spawn context for clean jax init.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(a.jobs, len(tasks)), maxtasksperchild=1) as pool:
        results = pool.map(_eval_one, tasks)
    out = {}
    for r in results:
        out.setdefault(r["cell"], {})[str(r["seed"])] = {
            k: r[k] for k in ("aep", "feasible", "min_dist_m", "boundary_penalty", "min_spacing")}
    tmp = a.out + ".tmp"
    json.dump({"cells": out, "walltime_s": round(time.time() - t0, 1)}, open(tmp, "w"), indent=2)
    os.replace(tmp, a.out)                       # atomic write (dispatch polls for this)
    print(f"wrote {a.out}: {len(results)} evals in {time.time()-t0:.0f}s "
          f"(jobs={min(a.jobs, len(tasks))})", flush=True)


if __name__ == "__main__":
    main()
