"""Run one uniform-rose cell × 7 ηT values × 3 sample seeds = 21 SGD runs.

Each LSF array task takes one cell (0..11), runs the full ηT sweep, and writes
its result JSON. The result JSON is then collected by the array-runner shell
script.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep \\
        pixi run python validation/stochastic_aep/run_uniform_per_cell.py \\
            --cell-idx <0..11> --out <out.json>
"""
import argparse
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


UNIFORM_CELLS = [
    f"results/matrix/problem_{farm}_n{n}_roseuniform.json"
    for farm in ("dei", "rowp")
    for n in (30, 40, 50, 60, 70, 80)
]
# η₀ = 50 m, so δ = ηT / 50
ETA_T_VALUES_M = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 25.0]
SAMPLE_SEEDS = [100000, 200000, 300000]


def one_run(cell_path, delta, sample_seed, init_seed=0, K=50, total_steps=8000):
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

        x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target,
                                          weights, wd, init_seed)
        bnd_j = jnp.array(boundary_local)
        bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))

        # 3 schedules: decay+ES at given δ, claude iter_192, gemini iter_192
        results = {}
        for sname, sched_builder, es_on in [
            (f"decay_es_eta_{ETA_T_VALUES_M[0]:.2f}", None, True),
            ("claude_iter192", funwake_iter192, False),
            ("gemini_iter192", gemini_iter192, False),
        ]:
            pass  # placeholder; we run only the decay baseline for this sweep

        # We only need to vary the baseline ηT here; Claude/Gemini are already in
        # matrix_fair for these uniform cells. Just run one delta baseline.
        sched = topfarm_default_decay(
            lr_init=50.0, gamma_min_factor=delta, total_steps=total_steps,
        )
        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, K,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=total_steps, init_seed=init_seed, sample_seed=sample_seed,
            early_stopping=True, es_threshold=0.1,
        )
        elapsed = time.time() - t0
        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"])
        return {
            "cell_path": cell_path, "delta": delta,
            "eta_t_m": 50.0 * delta,
            "sample_seed": sample_seed, "init_seed": init_seed,
            "K": K, "total_steps": total_steps,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "bp_final": bp_final, "sp_final": sp_final,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "cell_path": cell_path, "delta": delta, "sample_seed": sample_seed,
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

    cell = UNIFORM_CELLS[args.cell_idx]
    print(f"=== uniform cell {args.cell_idx}: {cell} ===", flush=True)

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"cell": cell, "runs": []}
    seen = {(r["delta"], r["sample_seed"]) for r in results["runs"]}

    tasks = [
        (50.0 * eta_t / 50.0, ss)  # delta = eta_t / 50, redundant but explicit
        for eta_t in ETA_T_VALUES_M
        for ss in SAMPLE_SEEDS
    ]
    # de-duplicate
    tasks = [(eta_t / 50.0, ss) for eta_t in ETA_T_VALUES_M for ss in SAMPLE_SEEDS]
    tasks = [(d, ss) for d, ss in tasks if (d, ss) not in seen]
    print(f"Tasks to run: {len(tasks)} (already done: {len(seen)})", flush=True)

    t_start = time.time()
    for i, (delta, ss) in enumerate(tasks, 1):
        r = one_run(cell, delta, ss, K=args.K, total_steps=args.total_steps)
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        tag = ("ERR " + r["error"][:80]) if "error" in r else (
            f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} elapsed={r['elapsed_s']}s"
        )
        print(f"[{i}/{len(tasks)}] δ={delta:.4f} (ηT={r.get('eta_t_m', 0):.2f}m) ss={ss}  {tag}",
              flush=True)
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
