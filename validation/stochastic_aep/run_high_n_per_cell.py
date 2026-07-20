"""High-N extension — one (cell × N) per task, runs 3 schedules × 7 ηT × 3 seeds.

Cells: 2 polygons × 4 roses × 2 N values (200, 300) = 16. Each task:
- decay+ES baseline at 7 ηT values × 3 sample seeds = 21 baseline runs
- claude iter_192 × 3 sample seeds = 3 runs
- gemini iter_192 × 3 sample seeds = 3 runs
= 27 runs per task.

LSF array [1-16].
"""
import argparse
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HIGH_N_CELLS = [
    f"results/matrix/problem_{farm}_n{n}_rose{rose}.json"
    for farm in ("dei", "rowp")
    for n in (200, 300)
    for rose in ("uniform", "omnidir", "dei", "rowp")
]
ETA_T_VALUES_M = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 25.0]
SAMPLE_SEEDS = [100000, 200000, 300000]


def one_run(cell_path, schedule_name, delta, sample_seed,
            init_seed=0, K=50, total_steps=8000):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from run_part3 import wind_aware_init, funwake_iter192
    from run_step3 import run_with_stochastic_schedule_es
    from run_step3_rowp import _translate_to_local, topfarm_default_decay
    from matrix_categorical_aep import categorical_rose_aep_factory, deterministic_full_rose_aep
    from run_matrix_stochastic import gemini_iter192

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

        if schedule_name == "decay_es":
            sched = topfarm_default_decay(
                lr_init=50.0, gamma_min_factor=delta, total_steps=total_steps,
            )
            es_on = True
        elif schedule_name == "claude_iter192":
            sched = funwake_iter192()
            es_on = False
        elif schedule_name == "gemini_iter192":
            sched = gemini_iter192()
            es_on = False
        else:
            raise ValueError(f"unknown schedule {schedule_name}")

        x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target,
                                          weights, wd, init_seed)
        bnd_j = jnp.array(boundary_local)
        bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))
        sp_init = float(spacing_penalty(jnp.array(x_init), jnp.array(y_init), float(min_spacing)))

        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, K,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=total_steps, init_seed=init_seed, sample_seed=sample_seed,
            early_stopping=es_on, es_threshold=0.1,
        )
        elapsed = time.time() - t0
        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"])
        return {
            "cell_path": cell_path, "schedule": schedule_name, "delta": delta,
            "eta_t_m": 50.0 * delta if schedule_name == "decay_es" else None,
            "sample_seed": sample_seed, "init_seed": init_seed,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "sp_init": sp_init,
            "bp_final": bp_final, "sp_final": sp_final,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "cell_path": cell_path, "schedule": schedule_name,
            "delta": delta, "sample_seed": sample_seed,
            "error": str(e)[:300],
            "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-idx", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    args = p.parse_args()

    cell = HIGH_N_CELLS[args.cell_idx]
    print(f"=== high-N cell {args.cell_idx}: {cell} ===", flush=True)

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"cell": cell, "runs": []}
    seen = {(r["schedule"], r.get("delta"), r["sample_seed"]) for r in results["runs"]}

    tasks = []
    # decay+ES at 7 ηT × 3 seeds
    for eta_t in ETA_T_VALUES_M:
        for ss in SAMPLE_SEEDS:
            d = eta_t / 50.0
            tasks.append(("decay_es", d, ss))
    # claude + gemini at 3 seeds each
    for ss in SAMPLE_SEEDS:
        tasks.append(("claude_iter192", None, ss))
        tasks.append(("gemini_iter192", None, ss))
    tasks = [t for t in tasks if t not in seen]
    print(f"Tasks: {len(tasks)} (already done: {len(seen)})", flush=True)

    t_start = time.time()
    for i, (sname, d, ss) in enumerate(tasks, 1):
        r = one_run(cell, sname, d, ss, K=args.K, total_steps=args.total_steps)
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        tag = ("ERR " + r["error"][:80]) if "error" in r else (
            f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} elapsed={r['elapsed_s']}s"
        )
        eta_str = f"ηT={50*d:.2f}m" if d else "      -"
        print(f"[{i}/{len(tasks)}] {sname:16s} {eta_str} ss={ss}  {tag}", flush=True)
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
