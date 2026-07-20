"""Step 3 — four-cell feasibility comparison: {sgd_baseline, iter_192} ×
{ES OFF, ES ON} on DEI training farm under stochastic AEP gradients
(Bastankhah k=0.04, K=50, 8000 iters, 20 restarts).

Reuses the Part-3 stochastic-objective skeleton (run_part3.py) and adds
Quick 2023 Algorithm 1 early-stopping in-loop: when ES is enabled and
lr_i/lr_0 <= threshold, the AEP gradient component is zeroed (j = alpha *
grad_con only). The fori_loop continues for total_steps; once grad_con == 0
the layout stops moving so the "break" is functionally a no-op (per
Algorithm 1, see REPORT_PART3 + Step-2 report).

Per-cell outputs:
- strict feasibility (bp == 0 & sp == 0) /20
- practical feasibility (bp < 1e-2 & sp < 1e-2) /20
- min pair distance range
- AEP (deterministic Weibull-marginalized) mean ± std

Optional 740-10 follow-up cells if --include-rowp.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
      validation/stochastic_aep/run_step3.py \\
      --restarts 20 --total-steps 8000 --K 50 \\
      --out validation/stochastic_aep/step3_result.json
"""
import argparse
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from pixwake.optim.sgd import boundary_penalty, spacing_penalty

from stochastic_aep import build_sim, deterministic_fine_grid_aep, stochastic_aep_factory
from run_part3 import (
    sgd_baseline,
    funwake_iter192,
    wind_aware_init,
    check_feasibility,
)


# ---------- Stochastic SGD with Quick 2023 Alg. 1 early-stopping ----------

def run_with_stochastic_schedule_es(
    schedule_fn, sim, aep_stoch_fn, K, n_target, boundary, min_spacing,
    weights, wd, total_steps=8000, init_seed=0, sample_seed=0,
    early_stopping=False, es_threshold=0.1,
):
    """Stochastic-objective Adam SGD with optional Quick 2023 Alg. 1 ES.

    When ES is enabled AND lr_i/lr_0 <= es_threshold, the AEP gradient is
    zeroed: search direction becomes alpha * grad_con. The fori_loop runs to
    completion; once grad_con == 0 at a feasible point, layout stops
    moving (equivalent to the Alg. 1 break — verified, see REPORT_STEP2).
    """
    boundary = jnp.array(boundary, dtype=jnp.float64)
    weights = jnp.array(weights, dtype=jnp.float64)

    def neg_aep(x, y, key):
        return -aep_stoch_fn(x, y, key, K)

    def con_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    grad_obj = jax.grad(neg_aep, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x, y = wind_aware_init(boundary, min_spacing, n_target, weights, wd, init_seed)

    # lr0, alpha0 — same recipe as Part 3
    k_init = jax.random.PRNGKey(sample_seed + 1_000_001)
    g_estimates_x, g_estimates_y = [], []
    for _ in range(5):
        k_init, sub = jax.random.split(k_init)
        gx_, gy_ = grad_obj(x, y, sub)
        g_estimates_x.append(gx_)
        g_estimates_y.append(gy_)
    gox = jnp.mean(jnp.stack(g_estimates_x), axis=0)
    goy = jnp.mean(jnp.stack(g_estimates_y), axis=0)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    es_enabled = bool(early_stopping)
    es_threshold = float(es_threshold)

    @jax.jit
    def run_loop(x, y, key0):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        eps = 1e-12

        def step(i, carry):
            x, y, mx, my, vx, vy, key = carry
            key, subkey = jax.random.split(key)

            lr, alpha, b1, b2 = schedule_fn(i, total_steps, lr0, alpha0)

            gox_i, goy_i = grad_obj(x, y, subkey)
            gcx, gcy = grad_con(x, y)

            # Quick 2023 Alg. 1: when ES active, drop AEP gradient.
            if es_enabled:
                lr_ratio = lr / lr0
                es_active = lr_ratio <= es_threshold
                zero_x = jnp.zeros_like(gox_i)
                zero_y = jnp.zeros_like(goy_i)
                gox_eff = jnp.where(es_active, zero_x, gox_i)
                goy_eff = jnp.where(es_active, zero_y, goy_i)
            else:
                gox_eff, goy_eff = gox_i, goy_i

            jx = gox_eff + alpha * gcx
            jy = goy_eff + alpha * gcy

            it = (i + 1).astype(float)
            mx_new = b1 * mx + (1 - b1) * jx
            my_new = b1 * my + (1 - b1) * jy
            vx_new = b2 * vx + (1 - b2) * jx**2
            vy_new = b2 * vy + (1 - b2) * jy**2

            mx_hat = mx_new / (1 - b1**it)
            my_hat = my_new / (1 - b1**it)
            vx_hat = vx_new / (1 - b2**it)
            vy_hat = vy_new / (1 - b2**it)

            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)

            return (x_new, y_new, mx_new, my_new, vx_new, vy_new, key)

        init = (x, y, mx, my, vx, vy, key0)
        final = jax.lax.fori_loop(0, total_steps, step, init)
        return final[0], final[1]

    key0 = jax.random.PRNGKey(sample_seed)
    return run_loop(x, y, key0)


def run_cell(label, schedule_fn, sim, aep_stoch_fn, problem, resource, args,
             early_stopping, es_threshold):
    boundary = problem["boundary_vertices"]
    min_spacing = float(problem["min_spacing_m"])
    n_target = int(problem["n_target"])
    weights = resource["sector_probability"]
    wd = resource["sector_centers_deg"]

    per_restart = []
    print(f"\n=== cell: {label}  early_stopping={early_stopping} threshold={es_threshold} ===", flush=True)
    for r in range(args.restarts):
        t0 = time.time()
        x_opt, y_opt = run_with_stochastic_schedule_es(
            schedule_fn, sim, aep_stoch_fn, args.K,
            n_target, boundary, min_spacing, weights, wd,
            total_steps=args.total_steps,
            init_seed=r, sample_seed=r + 100_000,
            early_stopping=early_stopping, es_threshold=es_threshold,
        )
        elapsed = time.time() - t0
        feas = check_feasibility(x_opt, y_opt, boundary, min_spacing)
        try:
            aep_det = deterministic_fine_grid_aep(
                sim, x_opt, y_opt, resource,
                ws_min=4.0, ws_max=25.0, ws_step=1.0, wd_step=1.0,
            )
        except Exception as e:
            aep_det = None
        entry = {
            "restart": r,
            "elapsed_s": round(elapsed, 1),
            "aep_gwh_det_weibull": aep_det,
            **feas,
        }
        per_restart.append(entry)
        tag = "✓" if feas["feasible"] else "✗"
        aep_str = f"{aep_det:.2f}" if aep_det is not None else "ERR"
        print(
            f"  r={r:2d} {elapsed:.0f}s {tag} AEP={aep_str}  "
            f"bp={feas['boundary_violation']:.2e} sp={feas['spacing_penalty']:.2e} "
            f"min_d={feas['min_pair_dist_m']:.0f}m",
            flush=True,
        )
    return per_restart


def summarize(per_restart):
    aeps = np.array([r["aep_gwh_det_weibull"] for r in per_restart
                     if r["aep_gwh_det_weibull"] is not None])
    bps = np.array([r["boundary_violation"] for r in per_restart])
    sps = np.array([r["spacing_penalty"] for r in per_restart])
    mins = np.array([r["min_pair_dist_m"] for r in per_restart])
    strict = sum(r["feasible"] for r in per_restart)
    practical = int(((bps < 1e-2) & (sps < 1e-2)).sum())
    return {
        "n": len(per_restart),
        "strict_feasibility": strict,
        "practical_feasibility": practical,
        "boundary_penalty_range": [float(bps.min()), float(bps.max())],
        "spacing_penalty_range": [float(sps.min()), float(sps.max())],
        "min_pair_dist_range_m": [float(mins.min()), float(mins.max())],
        "aep_mean": float(aeps.mean()) if len(aeps) else None,
        "aep_std": float(aeps.std(ddof=1)) if len(aeps) > 1 else None,
        "aep_best": float(aeps.max()) if len(aeps) else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="playground/problem.json")
    p.add_argument("--resource", default="validation/stochastic_aep/dei_weibull_12.json")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--wake-model", default="bastankhah_0.04")
    p.add_argument("--es-threshold", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--include-rowp", action="store_true", help="Run 740-10 cells too")
    p.add_argument("--skip-off", action="store_true", help="Skip ES OFF cells (reuse Part 3 data)")
    args = p.parse_args()

    with open(args.problem) as f:
        problem = json.load(f)
    with open(args.resource) as f:
        resource = json.load(f)

    sim, D = build_sim(problem, wake_model=args.wake_model)
    aep_stoch_fn = stochastic_aep_factory(sim, resource)

    schedules = {
        "sgd_baseline": sgd_baseline(),
        "funwake_iter192": funwake_iter192(),
    }

    t_start = time.time()
    cells = {}

    for sch_name, sched in schedules.items():
        for es_on, es_label in [(False, "OFF"), (True, "ON")]:
            if args.skip_off and not es_on:
                continue
            key = f"{sch_name}__ES_{es_label}"
            t_cell = time.time()
            per_restart = run_cell(
                key, sched, sim, aep_stoch_fn, problem, resource, args,
                early_stopping=es_on, es_threshold=args.es_threshold,
            )
            cells[key] = {
                "schedule": sch_name,
                "early_stopping": es_on,
                "es_threshold": args.es_threshold,
                "cell_elapsed_s": round(time.time() - t_cell, 1),
                "per_restart": per_restart,
                "summary": summarize(per_restart),
            }
            print(f"   ↳ summary: {cells[key]['summary']}", flush=True)

    rowp_cells = {}
    if args.include_rowp:
        rowp_problem = json.load(open("validation/stochastic_aep/problem_740.json"))
        rowp_resource = json.load(open("validation/stochastic_aep/rowp_weibull_12.json"))
        sim_r, _ = build_sim(rowp_problem, wake_model="noj_0.05")
        aep_r = stochastic_aep_factory(sim_r, rowp_resource)
        for sch_name, sched in schedules.items():
            for es_on, es_label in [(False, "OFF"), (True, "ON")]:
                key = f"ROWP__{sch_name}__ES_{es_label}"
                t_cell = time.time()
                boundary_r = rowp_problem["boundary_vertices"]
                min_spacing_r = float(rowp_problem["min_spacing_m"])
                n_target_r = int(rowp_problem["n_target"])
                weights_r = rowp_resource["sector_probability"]
                wd_r = rowp_resource["sector_centers_deg"]
                pr = []
                for r in range(args.restarts):
                    t0 = time.time()
                    x_opt, y_opt = run_with_stochastic_schedule_es(
                        sched, sim_r, aep_r, args.K,
                        n_target_r, boundary_r, min_spacing_r,
                        weights_r, wd_r, total_steps=args.total_steps,
                        init_seed=r, sample_seed=r + 200_000,
                        early_stopping=es_on, es_threshold=args.es_threshold,
                    )
                    elapsed = time.time() - t0
                    feas = check_feasibility(x_opt, y_opt, boundary_r, min_spacing_r)
                    try:
                        aep_det = deterministic_fine_grid_aep(
                            sim_r, x_opt, y_opt, rowp_resource,
                            ws_min=4.0, ws_max=25.0, ws_step=1.0, wd_step=1.0,
                        )
                    except Exception:
                        aep_det = None
                    pr.append({
                        "restart": r, "elapsed_s": round(elapsed, 1),
                        "aep_gwh_det_weibull": aep_det, **feas,
                    })
                    print(
                        f"  ROWP {sch_name}/ES_{es_label} r={r:2d} {elapsed:.0f}s "
                        f"{'✓' if feas['feasible'] else '✗'} "
                        f"AEP={aep_det:.2f}  bp={feas['boundary_violation']:.2e}",
                        flush=True,
                    )
                rowp_cells[key] = {
                    "schedule": sch_name, "early_stopping": es_on,
                    "es_threshold": args.es_threshold, "wake": "noj_0.05",
                    "cell_elapsed_s": round(time.time() - t_cell, 1),
                    "per_restart": pr, "summary": summarize(pr),
                }
                print(f"   ↳ ROWP summary: {rowp_cells[key]['summary']}", flush=True)

    out = {
        "config": vars(args),
        "elapsed_total_s": round(time.time() - t_start, 1),
        "dei_cells": cells,
        "rowp_cells": rowp_cells,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== STEP 3 SUMMARIES ===")
    for k, c in {**cells, **rowp_cells}.items():
        print(f"{k}: {c['summary']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
