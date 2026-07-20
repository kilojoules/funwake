"""Task J — α-weight ablation under K=50 stochastic gradient.

Replaces tools/alpha_ablation.py (which used coarse deterministic gradient
via tools/run_optimizer.py). Same FACTORS, same problems, paper-faithful
K=50 categorical-rose MC inner gradient.

Schedule: iter_192 with α multiplied by FACTOR (LR/bumps/β unchanged).
Problems: playground/problem.json (DEI train) + results/problem_rowp.json
(ROWP held-out).
Seeds: 3 sample seeds per (factor, problem).
Total runs: 9 × 2 × 3 = 54. Wall ≈ 2.5–3 hr.

Resume-safe: writes incrementally to --out JSON.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_j_alphaweight.py \\
        --out validation/stochastic_aep/task_j_alpha_k50.json
"""
import argparse
import json
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FACTORS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
PROBLEMS = [
    ("dei_train", "playground/problem.json",      "bastankhah_0.04"),
    # Paper's deployment regime: harness.py uses Bastankhah k=0.04
    # unconditionally — so paper's 4271.5 GWh figure for iter_192 ROWP is
    # Bastankhah-on-ROWP-polygon, not the physically-correct NOJ-on-740-10
    # (which lives in run_step3_rowp.py for the 740-10 validation work).
    # To make alpha-ablation comparable to the paper, use Bastankhah here.
    ("rowp_held", "results/problem_rowp.json",    "bastankhah_0.04"),
]
SEEDS = (100000, 200000, 300000)


def one_run(problem_path, wake_model, factor, sample_seed, K, total_steps,
            translate_rowp):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from matrix_categorical_aep import (
        categorical_rose_aep_factory,
        deterministic_full_rose_aep,
    )
    from run_step3 import run_with_stochastic_schedule_es
    from run_step3_rowp import _translate_to_local
    from schedules_ablation import funwake_iter192_alpha_scaled

    try:
        with open(os.path.join(PROJECT_ROOT, problem_path)) as f:
            problem = json.load(f)
        sim, D = build_sim(problem, wake_model=wake_model)
        aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])

        if translate_rowp:
            boundary_local, _, _ = _translate_to_local(problem["boundary_vertices"])
        else:
            boundary_local = problem["boundary_vertices"]

        n_target = int(problem["n_target"])
        min_spacing = float(problem["min_spacing_m"])
        weights = problem["wind_rose"]["weights"]
        wd = problem["wind_rose"]["directions_deg"]

        sched = funwake_iter192_alpha_scaled(lr_init=50.0, factor=factor)

        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            sched, sim, aep_stoch_fn, K,
            n_target, boundary_local, min_spacing, weights, wd,
            total_steps=total_steps, init_seed=0, sample_seed=sample_seed,
            early_stopping=False, es_threshold=0.1,
        )
        elapsed = time.time() - t0

        x_arr = np.asarray(x_opt); y_arr = np.asarray(y_opt)
        bnd_j = jnp.array(boundary_local)
        bp_final = float(boundary_penalty(jnp.array(x_arr), jnp.array(y_arr), bnd_j))
        sp_final = float(spacing_penalty(jnp.array(x_arr), jnp.array(y_arr), min_spacing))
        aep_det = deterministic_full_rose_aep(
            sim, jnp.array(x_arr), jnp.array(y_arr), problem["wind_rose"]
        )
        feasible = (bp_final < 1e-2) and (sp_final < 1e-2)
        return {
            "factor": factor, "problem": problem_path, "wake_model": wake_model,
            "sample_seed": sample_seed,
            "aep_det_gwh": float(aep_det),
            "bp_final": bp_final, "sp_final": sp_final, "feasible": bool(feasible),
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            "factor": factor, "problem": problem_path, "sample_seed": sample_seed,
            "error": str(e)[:300], "trace": traceback.format_exc()[:500],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    args = p.parse_args()

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"K": args.K, "total_steps": args.total_steps, "runs": []}
    seen = {(r.get("factor"), r.get("problem"), r.get("sample_seed"))
            for r in results["runs"]}

    tasks = []
    for label, problem_path, wake_model in PROBLEMS:
        translate_rowp = (label == "rowp_held")
        for factor in FACTORS:
            for ss in SEEDS:
                key = (factor, problem_path, ss)
                if key not in seen:
                    tasks.append((problem_path, wake_model, factor, ss, translate_rowp))
    print(f"Tasks: {len(tasks)} (already done: {len(seen)})", flush=True)

    t_start = time.time()
    for i, (problem_path, wake_model, factor, ss, translate_rowp) in enumerate(tasks, 1):
        r = one_run(problem_path, wake_model, factor, ss,
                    args.K, args.total_steps, translate_rowp)
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if "error" in r:
            tag = f"ERR {r['error'][:80]}"
        else:
            feas = "FEAS" if r["feasible"] else "INFEAS"
            tag = (f"AEP={r['aep_det_gwh']:.2f} bp={r['bp_final']:.1e} "
                   f"sp={r['sp_final']:.1e} {feas} elapsed={r['elapsed_s']}s")
        print(f"[{i}/{len(tasks)}] factor={factor:.2f} {problem_path:38s} "
              f"ss={ss}  {tag}", flush=True)

    elapsed_min = (time.time() - t_start) / 60
    print(f"\nWall: {elapsed_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
