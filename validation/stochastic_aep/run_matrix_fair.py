"""Fair-baseline 48-cell matrix re-eval.

Replaces the prior matrix's constant-LR `sgd_baseline` (Step 3 established as
denuded — can't engage ES, fails feasibility under stochastic gradients) with
the PROPERLY-EQUIPPED `topfarm_default_decay + ES` baseline from the ROWP
Step-3 run.

Per (cell, schedule), runs ≥3 sample seeds so per-cell spread can be measured
empirically rather than borrowed from the prior 0.022% noise floor.

Cells (48) × {decay_es_baseline, claude_iter192, gemini_iter192} × 3 seeds =
432 SGD runs.

Stochastic K=50 categorical-rose sampling (matrix_categorical_aep.py).
total_steps = 8000. init_seed = 0. Same plumbing fixes (CCW, translation) as
the original matrix re-eval.

Parallel execution via ProcessPoolExecutor (per-worker JAX is independent).
Resume-safe via incremental JSON writes per result.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_matrix_fair.py \\
        --workers 4 --sample-seeds 100000 200000 300000 \\
        --out validation/stochastic_aep/matrix_fair.json
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


def run_task(cell_path, schedule_name, sample_seed, init_seed, K, total_steps,
             es_enabled, es_threshold):
    """Run one (cell, schedule, sample_seed) combination. Called inside a
    subprocess so JAX state stays isolated."""
    import jax
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    # Path setup inside subprocess
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from run_part3 import wind_aware_init, funwake_iter192, sgd_baseline
    from run_step3 import run_with_stochastic_schedule_es
    from run_step3_rowp import _translate_to_local, topfarm_default_decay, preflight_es
    from matrix_categorical_aep import (
        categorical_rose_aep_factory,
        deterministic_full_rose_aep,
    )
    from run_matrix_stochastic import gemini_iter192

    SCHEDULE_BUILDERS = {
        "claude_iter192": funwake_iter192,
        "gemini_iter192": gemini_iter192,
        "decay_es_baseline": lambda: topfarm_default_decay(
            lr_init=50.0, gamma_min_factor=0.01, total_steps=total_steps,
        ),
        "sgd_baseline_const_lr": sgd_baseline,  # reference only
    }

    try:
        with open(os.path.join(PROJECT_ROOT, cell_path)) as f:
            problem = json.load(f)
        sim, D = build_sim(problem, wake_model="bastankhah_0.04")
        aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])

        boundary_local, ox, oy = _translate_to_local(problem["boundary_vertices"])
        n_target = int(problem["n_target"])
        min_spacing = float(problem["min_spacing_m"])
        weights = problem["wind_rose"]["weights"]
        wd = problem["wind_rose"]["directions_deg"]

        x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target,
                                          weights, wd, init_seed)
        bnd_j = jnp.array(boundary_local)
        bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))
        sp_init = float(spacing_penalty(jnp.array(x_init), jnp.array(y_init), float(min_spacing)))

        sched = SCHEDULE_BUILDERS[schedule_name]()
        es_trigger = None
        if hasattr(sched, "lr_trajectory"):
            es_trigger = preflight_es(sched, total_steps, es_threshold)

        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, K,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=total_steps, init_seed=init_seed, sample_seed=sample_seed,
            early_stopping=es_enabled, es_threshold=es_threshold,
        )
        elapsed = time.time() - t0

        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"])

        # min pair distance
        dx = x_arr[:, None] - x_arr[None, :]
        dy = y_arr[:, None] - y_arr[None, :]
        d2 = dx**2 + dy**2 + np.eye(len(x_arr)) * 1e30
        min_d = float(np.sqrt(d2.min()))

        return {
            "cell_path": cell_path,
            "schedule": schedule_name,
            "sample_seed": sample_seed,
            "es_enabled": es_enabled,
            "es_threshold": es_threshold,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init,
            "sp_init": sp_init,
            "bp_final": bp_final,
            "sp_final": sp_final,
            "min_pair_dist_m": min_d,
            "elapsed_s": round(elapsed, 1),
            "es_trigger": es_trigger,
        }
    except Exception as e:
        return {
            "cell_path": cell_path,
            "schedule": schedule_name,
            "sample_seed": sample_seed,
            "error": str(e)[:300],
            "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/matrix/manifest.json")
    p.add_argument("--max-n", type=int, default=80)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--sample-seeds", type=int, nargs="+",
                   default=[100000, 200000, 300000])
    p.add_argument("--es-threshold", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--schedules", nargs="+",
                   default=["decay_es_baseline", "claude_iter192", "gemini_iter192"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest = json.load(open(os.path.join(PROJECT_ROOT, args.manifest)))
    cells = [c for c in manifest["cells"] if c["n"] <= args.max_n]
    print(f"Cells: {len(cells)}  Schedules: {args.schedules}  "
          f"Seeds/cell: {args.sample_seeds}  Workers: {args.workers}",
          flush=True)

    # Load existing for resume
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"config": vars(args), "runs": []}
    seen = {(r["cell_path"], r["schedule"], r["sample_seed"]) for r in results["runs"]}

    tasks = []
    for cell in cells:
        for sch in args.schedules:
            for ss in args.sample_seeds:
                if (cell["path"], sch, ss) in seen:
                    continue
                es_on = (sch == "decay_es_baseline")
                tasks.append((
                    cell["path"], sch, ss, args.init_seed, args.K, args.total_steps,
                    es_on, args.es_threshold,
                ))

    print(f"Tasks to run: {len(tasks)} (already done: {len(seen)})", flush=True)
    t_start = time.time()
    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(run_task, *t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            results["runs"].append(r)
            # Incremental save
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            done += 1
            tag = ("ERR " + r["error"][:80]) if "error" in r else (
                f"AEP={r['aep_det_gwh']:.2f} bp_init={r['bp_init']:.1e} "
                f"bp_final={r['bp_final']:.1e} min_d={r['min_pair_dist_m']:.0f}m "
                f"elapsed={r['elapsed_s']}s"
            )
            print(f"[{done}/{total}] {r['cell_path'].split('/')[-1]:42s} | "
                  f"{r['schedule']:18s} ss={r['sample_seed']}  {tag}",
                  flush=True)

    results["elapsed_total_s"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {results['elapsed_total_s']/60:.1f} min wall. Wrote {args.out}")


if __name__ == "__main__":
    main()
