"""Experiment A (δ-sweep baseline) + Experiment B (iter_192 ES on/off with
lr_ratio trace) on the 18 headline cells.

Headline cells = multidirectional roses (omnidir, dei, rowp) × N ∈ {60, 70, 80} ×
2 polygons (DEI, ROWP) = 18 cells.

Experiment A — δ-sweep:
  For each cell, run the TopFarm-default decay+ES baseline at multiple
  gamma_min_factor (= δ) values. Default 0.01 is already in matrix_fair.json;
  this runs 6 NEW δ values ∈ {0.5, 0.1, 0.05, 0.02, 0.005, 0.001} × 3 seeds.

Experiment B — iter_192 + ES with lr_ratio trace:
  Run iter_192 with ES enabled (threshold 0.1) at 3 seeds per cell. Dump the
  full per-step lr_ratio trajectory (deterministic per schedule) so ES-firing
  time/whether is auditable for the non-monotonic schedule. iter_192 ES-off is
  already in matrix_fair.json; only ES-on is added here.

Total new compute: 18 × 6 × 3 = 324 (A) + 18 × 1 × 3 = 54 (B) = 378 runs.

Usage:
    PYTHONPATH=playground/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_delta_sweep_and_es.py \\
        --workers 1 \\
        --out validation/stochastic_aep/delta_sweep_and_es.json
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


HEADLINE_CELLS = [
    f"results/matrix/problem_{farm}_n{n}_rose{rose}.json"
    for farm in ("dei", "rowp")
    for n in (60, 70, 80)
    for rose in ("omnidir", "dei", "rowp")
]
DELTA_VALUES_NEW = [0.5, 0.1, 0.05, 0.02, 0.005, 0.001]  # 0.01 already done


def run_task(cell_path, task_kind, sample_seed, init_seed, K, total_steps,
             delta=None, dump_lr_trace=False):
    """Run one task. task_kind ∈ {'delta_baseline', 'iter192_es_on'}.

    Returns dict with aep, feasibility, lr_ratio trajectory (if requested)."""
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "playground/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from run_part3 import wind_aware_init, funwake_iter192
    from run_step3 import run_with_stochastic_schedule_es
    from run_step3_rowp import _translate_to_local, topfarm_default_decay, preflight_es
    from matrix_categorical_aep import categorical_rose_aep_factory, deterministic_full_rose_aep

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

        if task_kind == "delta_baseline":
            sched = topfarm_default_decay(
                lr_init=50.0, gamma_min_factor=delta, total_steps=total_steps,
            )
            es_enabled = True
            es_pf = preflight_es(sched, total_steps, 0.1)
        elif task_kind == "iter192_es_on":
            sched = funwake_iter192()
            es_enabled = True
            es_pf = None
        else:
            raise ValueError(f"unknown task_kind {task_kind}")

        # Pre-compute lr_ratio trajectory for the trace (deterministic in step)
        lr_ratio_trajectory = None
        es_first_cross = None
        if dump_lr_trace:
            # Sample the schedule at K_trace points across [0, total_steps)
            n_pts = 200
            steps_sample = np.linspace(0, total_steps - 1, n_pts).astype(int)
            lr_vals = []
            for s in steps_sample:
                lr_v, _, _, _ = sched(int(s), total_steps, 50.0, 1.0)
                lr_vals.append(float(lr_v))
            lr_arr = np.asarray(lr_vals)
            # iter_192 schedule's lr ranges from 0 (warmup) to lr_peak=200; use
            # the running maximum as the reference (otherwise warmup gives a
            # trivially-small lr_ratio at iter 0).
            lr_running_max = np.maximum.accumulate(lr_arr)
            lr_ratio = lr_arr / np.maximum(lr_running_max, 1e-10)
            lr_ratio_trajectory = {
                "steps_sampled": steps_sample.tolist(),
                "lr_values": lr_arr.tolist(),
                "lr_ratio_to_running_max": lr_ratio.tolist(),
                "lr_running_max": lr_running_max.tolist(),
            }
            # ES-firing definition matches the impl: lr_i / lr_0 ≤ threshold.
            # lr_0 = the lr at step 0 (which for iter_192 is 0 — schedule starts
            # at warmup start; for decay schedule = lr_init). Use the *true*
            # implementation's check: lr_at_step / lr_init=50.0
            lr_ratio_to_lr_init = lr_arr / 50.0
            crosses = lr_ratio_to_lr_init <= 0.1
            es_first_cross = int(np.argmax(crosses)) if crosses.any() else None
            lr_ratio_trajectory["lr_ratio_to_lr_init_step0"] = lr_ratio_to_lr_init.tolist()
            lr_ratio_trajectory["es_first_cross_idx"] = es_first_cross
            lr_ratio_trajectory["es_first_cross_step"] = (
                int(steps_sample[es_first_cross]) if es_first_cross is not None else None
            )
            lr_ratio_trajectory["fires_at_some_point"] = bool(crosses.any())
            lr_ratio_trajectory["fires_at_iter_0_warmup"] = bool(crosses[0])

        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, K,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=total_steps, init_seed=init_seed, sample_seed=sample_seed,
            early_stopping=es_enabled, es_threshold=0.1,
        )
        elapsed = time.time() - t0

        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"])
        dx = x_arr[:, None] - x_arr[None, :]
        dy = y_arr[:, None] - y_arr[None, :]
        d2 = dx**2 + dy**2 + np.eye(len(x_arr)) * 1e30
        min_d = float(np.sqrt(d2.min()))

        return {
            "cell_path": cell_path,
            "task_kind": task_kind,
            "sample_seed": sample_seed,
            "delta": delta,
            "es_enabled": es_enabled,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "sp_init": sp_init,
            "bp_final": bp_final, "sp_final": sp_final,
            "min_pair_dist_m": min_d,
            "elapsed_s": round(elapsed, 1),
            "es_preflight": es_pf,
            "lr_ratio_trajectory": lr_ratio_trajectory,
        }
    except Exception as e:
        return {
            "cell_path": cell_path,
            "task_kind": task_kind,
            "sample_seed": sample_seed,
            "delta": delta,
            "error": str(e)[:300],
            "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--sample-seeds", type=int, nargs="+",
                   default=[100000, 200000, 300000])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load existing for resume
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"config": vars(args), "runs": []}
    seen = {(r["cell_path"], r["task_kind"], r["sample_seed"], r.get("delta"))
            for r in results["runs"]}

    # Build task list
    tasks = []
    # Experiment A: delta-sweep baseline (only NEW deltas, default 0.01 reused)
    for cell_path in HEADLINE_CELLS:
        for delta in DELTA_VALUES_NEW:
            for ss in args.sample_seeds:
                if (cell_path, "delta_baseline", ss, delta) in seen:
                    continue
                tasks.append((
                    cell_path, "delta_baseline", ss, args.init_seed,
                    args.K, args.total_steps, delta, False,
                ))
    # Experiment B: iter_192 ES-on with lr_ratio trace (dump trace on seed 100000)
    for cell_path in HEADLINE_CELLS:
        for ss in args.sample_seeds:
            if (cell_path, "iter192_es_on", ss, None) in seen:
                continue
            tasks.append((
                cell_path, "iter192_es_on", ss, args.init_seed,
                args.K, args.total_steps, None,
                ss == args.sample_seeds[0],  # only trace once per cell
            ))

    print(f"Headline cells: {len(HEADLINE_CELLS)}",
          f"δ values (NEW): {DELTA_VALUES_NEW}",
          f"Sample seeds: {args.sample_seeds}",
          f"Workers: {args.workers}",
          f"Tasks: {len(tasks)} (already done: {len(seen)})",
          sep="\n", flush=True)

    t_start = time.time()
    done = 0
    total = len(tasks)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                  mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(run_task, *t) for t in tasks]
            for fut in as_completed(futures):
                r = fut.result()
                results["runs"].append(r)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                done += 1
                tag = ("ERR " + r["error"][:80]) if "error" in r else (
                    f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} "
                    f"elapsed={r['elapsed_s']}s"
                )
                print(f"[{done}/{total}] {r['cell_path'].split('/')[-1]:42s} "
                      f"| {r['task_kind']:18s} δ={r.get('delta')} ss={r['sample_seed']}  {tag}",
                      flush=True)
    else:
        for t in tasks:
            r = run_task(*t)
            results["runs"].append(r)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            done += 1
            tag = ("ERR " + r["error"][:80]) if "error" in r else (
                f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} "
                f"elapsed={r['elapsed_s']}s"
            )
            print(f"[{done}/{total}] {r['cell_path'].split('/')[-1]:42s} "
                  f"| {r['task_kind']:18s} δ={r.get('delta')} ss={r['sample_seed']}  {tag}",
                  flush=True)

    results["elapsed_total_s"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {results['elapsed_total_s']/60:.1f} min wall. Wrote {args.out}")


if __name__ == "__main__":
    main()
