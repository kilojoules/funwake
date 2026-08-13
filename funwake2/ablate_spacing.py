"""Constraint-form ablation: does a QUADRATIC spacing penalty fix the uniform-wind
feasibility failures that the LINEAR (exact-penalty, TopFarm-standard) form shows?

Runs schedules on farms x seeds under three spacing-penalty forms, monkeypatched per
worker (nothing in the codebase is permanently changed). The feasibility CHECK is
form-independent (d >= min_spacing); only the penalty DRIVING optimization changes.

  linear         : sum(max(0, min_spacing^2 - d^2))        (current; exact/L1 in d^2)
  quadratic       : sum(max(0, min_spacing^2 - d^2)^2)      (naive switch; magnitude explodes)
  quadratic_norm  : sum(max(0, min_spacing^2 - d^2)^2)/min_spacing^2  (scale-matched at deep viol)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _make_quad(norm):
    import jax.numpy as jnp

    def _quad(x, y, min_spacing, rho=100.0):
        n = x.shape[0]
        if n < 2:
            return jnp.array(0.0)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dsq = dx**2 + dy**2
        iu, ju = jnp.triu_indices(n, k=1)
        pair = dsq[iu, ju]
        viol = jnp.maximum(0.0, min_spacing**2 - pair)
        p = jnp.sum(viol**2)
        return p / (min_spacing**2) if norm else p
    return _quad


def _eval_one(task):
    cell, seed, sched, steps, form = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import evaluator as E
    import skeleton_v2
    try:
        import matrix_cells
        matrix_cells.register(E)
    except Exception:
        pass
    if form == "quadratic":
        skeleton_v2.spacing_penalty = _make_quad(norm=False)
    elif form == "quadratic_norm":
        skeleton_v2.spacing_penalty = _make_quad(norm=True)
    # form == "linear": leave the imported (pixwake) spacing_penalty as-is
    fn = E.load_schedule(sched)
    r = E.evaluate(cell, fn, seed=seed, total_steps=steps, gamma_min=0.01)
    return {"cell": cell, "seed": int(seed), "form": form,
            "sched": os.path.basename(sched), "aep": float(r["aep_gwh"]),
            "feasible": bool(r["feasible"]), "min_dist_m": float(r.get("min_dist_m", 0.0) or 0.0),
            "min_spacing": float(r.get("min_spacing", 0.0) or 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--forms", nargs="+", default=["linear", "quadratic", "quadratic_norm"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    tasks = [(c, s, os.path.abspath(sc), a.steps, f)
             for sc in a.schedules for c in a.cells for s in a.seeds for f in a.forms]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(a.jobs, len(tasks)), maxtasksperchild=1) as pool:
        results = pool.map(_eval_one, tasks)
    json.dump({"results": results}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {len(results)} evals", flush=True)


if __name__ == "__main__":
    main()
