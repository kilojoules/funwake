"""Task F — baseline ES-cost per cell: baseline ES-on + ES-off at best-ηT, 3 seeds.

LSF array task: cell_idx ∈ 0..29 → multidir cell (3 roses × 2 polygons × 5 N).
Per cell: 3 sample seeds × 2 ES modes = 6 SGD runs.

Resume-safe: writes one JSON per cell.

Usage (local or on gbar):
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_f_es_cost_per_cell.py \\
        --cell-idx <0..29> --out <out.json>
"""
import argparse
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# (cell_path_basename, best_eta_t_m) — best-ηT from existing delta_sweep
# (N=60-80) and high-N analysis (N=200/300) per cell.
CELLS_AND_BEST_ETA_T = [
    # N=60 multidir (6)
    ("dei_n60_roseomnidir",  5.0),   # from delta_sweep best_delta=0.1
    ("dei_n60_rosedei",      2.5),   # best_delta=0.05
    ("dei_n60_roserowp",     0.5),   # best_delta=0.01
    ("rowp_n60_roseomnidir", 0.5),   # best_delta=0.01
    ("rowp_n60_rosedei",     0.25),  # best_delta=0.005
    ("rowp_n60_roserowp",    5.0),   # best_delta=0.1
    # N=70 multidir (6)
    ("dei_n70_roseomnidir",  2.5),
    ("dei_n70_rosedei",      25.0),  # best_delta=0.5
    ("dei_n70_roserowp",     5.0),
    ("rowp_n70_roseomnidir", 5.0),
    ("rowp_n70_rosedei",     1.0),
    ("rowp_n70_roserowp",    1.0),
    # N=80 multidir (6)
    ("dei_n80_roseomnidir",  5.0),
    ("dei_n80_rosedei",      5.0),
    ("dei_n80_roserowp",     5.0),
    ("rowp_n80_roseomnidir", 1.0),
    ("rowp_n80_rosedei",     1.0),
    ("rowp_n80_roserowp",    5.0),
    # N=200 multidir (6) — from high-N best-ηT
    ("dei_n200_roseomnidir", 2.5),
    ("dei_n200_rosedei",     5.0),
    ("dei_n200_roserowp",    5.0),
    ("rowp_n200_roseomnidir", 0.5),
    ("rowp_n200_rosedei",    0.5),
    ("rowp_n200_roserowp",   5.0),
    # N=300 multidir (6) — from high-N best-ηT
    ("dei_n300_roseomnidir", 5.0),
    ("dei_n300_rosedei",     1.0),
    ("dei_n300_roserowp",    5.0),
    ("rowp_n300_roseomnidir", 25.0),
    ("rowp_n300_rosedei",    25.0),
    ("rowp_n300_roserowp",   5.0),
]


def one_run(cell_path, eta_t_m, sample_seed, es_on, init_seed=0, K=50, total_steps=8000):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from run_part3 import wind_aware_init
    from run_step3 import run_with_stochastic_schedule_es
    from run_step3_rowp import _translate_to_local, topfarm_default_decay
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

        sched = topfarm_default_decay(
            lr_init=50.0, gamma_min_factor=eta_t_m / 50.0, total_steps=total_steps,
        )
        x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target,
                                          weights, wd, init_seed)
        bnd_j = jnp.array(boundary_local)
        bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))

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
            "cell_path": cell_path, "eta_t_m": eta_t_m,
            "es_on": es_on, "sample_seed": sample_seed, "init_seed": init_seed,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "bp_final": bp_final, "sp_final": sp_final,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "cell_path": cell_path, "eta_t_m": eta_t_m, "es_on": es_on,
            "sample_seed": sample_seed,
            "error": str(e)[:300], "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-idx", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    args = p.parse_args()

    name, eta_t = CELLS_AND_BEST_ETA_T[args.cell_idx]
    cell = f"results/matrix/problem_{name}.json"
    print(f"=== Task F cell {args.cell_idx}: {name} ηT={eta_t}m ===", flush=True)

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"cell": cell, "eta_t_m": eta_t, "runs": []}
    seen = {(r["es_on"], r["sample_seed"]) for r in results["runs"]}

    tasks = []
    for ss in (100000, 200000, 300000):
        for es_on in (False, True):
            if (es_on, ss) not in seen:
                tasks.append((ss, es_on))
    print(f"Tasks: {len(tasks)} (already done: {len(seen)})", flush=True)

    t_start = time.time()
    for i, (ss, es_on) in enumerate(tasks, 1):
        r = one_run(cell, eta_t, ss, es_on, K=args.K, total_steps=args.total_steps)
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        tag = ("ERR " + r["error"][:80]) if "error" in r else (
            f"AEP={r['aep_det_gwh']:.2f} bp={r['bp_final']:.1e} elapsed={r['elapsed_s']}s"
        )
        print(f"[{i}/{len(tasks)}] ES={'on' if es_on else 'off'} ss={ss}  {tag}", flush=True)
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
