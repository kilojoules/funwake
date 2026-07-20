"""Task 2 — β-sweep on 3 lowest-margin cells.

For each (cell, best_eta_t) pair: run the decay+ES baseline at
β = (0.3, 0.5) — iter_192's betas — and at (0.1, 0.2) — TopFarm default —
× 3 sample seeds. Confirm ES fires and bp_final = 0 at both β settings.

The (0.1, 0.2) runs duplicate hardening data for sanity but at the SAME
init seed = 0, K, total_steps so they're directly paired.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_beta_sweep.py \\
        --cells <p1> <p2> <p3> --eta-t-vals <e1> <e2> <e3> \\
        --betas 0.1,0.2 0.3,0.5 \\
        --out validation/stochastic_aep/beta_sweep.json
"""
import argparse
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def topfarm_decay_with_betas(lr_init, gamma_min_factor, total_steps,
                              beta1, beta2):
    """topfarm_default_decay variant with overridable betas."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    from run_step3_rowp import _bisect_mid
    import jax.numpy as jnp
    import numpy as np
    gamma_min = lr_init * gamma_min_factor
    mid = _bisect_mid(lr_init, gamma_min, total_steps)
    ts = np.arange(1, total_steps + 1, dtype=np.float64)
    factors = 1.0 / (1.0 + mid * ts)
    lr_array = lr_init * np.cumprod(factors)
    lr_array_j = jnp.asarray(lr_array, dtype=jnp.float64)

    def apply(step, total_steps_arg, lr0_arg, alpha0_arg):
        lr = lr_array_j[step]
        alpha = alpha0_arg * lr_init / jnp.maximum(lr, 1e-10)
        return lr, alpha, jnp.float64(beta1), jnp.float64(beta2)
    apply.lr_trajectory = lr_array
    apply.mid = mid
    apply.lr_init = lr_init
    apply.gamma_min = gamma_min
    return apply


def run_one(cell_path, eta_t_m, beta1, beta2, sample_seed,
            init_seed=0, K=50, total_steps=8000):
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
    from run_step3_rowp import _translate_to_local
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

        sched = topfarm_decay_with_betas(
            lr_init=50.0, gamma_min_factor=eta_t_m / 50.0,
            total_steps=total_steps, beta1=beta1, beta2=beta2,
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
            early_stopping=True, es_threshold=0.1,
        )
        elapsed = time.time() - t0
        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"])
        return {
            "cell_path": cell_path, "eta_t_m": eta_t_m,
            "beta1": beta1, "beta2": beta2,
            "sample_seed": sample_seed, "init_seed": init_seed,
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "bp_final": bp_final, "sp_final": sp_final,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "cell_path": cell_path, "eta_t_m": eta_t_m, "beta1": beta1, "beta2": beta2,
            "sample_seed": sample_seed,
            "error": str(e)[:300], "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", required=True,
                   help="Cell problem JSON paths (one per low-margin cell)")
    p.add_argument("--eta-t-vals", nargs="+", type=float, required=True,
                   help="Best-ηT value per cell (paired with --cells)")
    p.add_argument("--betas", nargs="+", default=["0.1,0.2", "0.3,0.5"],
                   help="Comma-separated beta1,beta2 pairs")
    p.add_argument("--sample-seeds", type=int, nargs="+",
                   default=[100000, 200000, 300000])
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if len(args.cells) != len(args.eta_t_vals):
        raise SystemExit("--cells and --eta-t-vals must have same length")
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"config": vars(args), "runs": []}
    seen = {(r["cell_path"], r["eta_t_m"], r["beta1"], r["beta2"], r["sample_seed"])
            for r in results["runs"]}

    betas = []
    for b in args.betas:
        b1, b2 = b.split(",")
        betas.append((float(b1), float(b2)))

    tasks = []
    for cell, eta_t in zip(args.cells, args.eta_t_vals):
        for b1, b2 in betas:
            for ss in args.sample_seeds:
                tasks.append((cell, eta_t, b1, b2, ss))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[4]) not in seen]

    print(f"Tasks to run: {len(tasks)}", flush=True)
    t_start = time.time()
    for i, (cell, eta_t, b1, b2, ss) in enumerate(tasks, 1):
        r = run_one(cell, eta_t, b1, b2, ss, K=args.K, total_steps=args.total_steps)
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        tag = ("ERR " + r["error"][:80]) if "error" in r else (
            f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} elapsed={r['elapsed_s']}s"
        )
        print(f"[{i}/{len(tasks)}] {os.path.basename(cell)[:30]:30s} "
               f"ηT={eta_t:.2f}m β=({b1},{b2}) ss={ss}  {tag}",
               flush=True)
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
