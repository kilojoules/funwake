"""Part 3 — confound gate: iter_192 vs sgd_baseline under stochastic gradients.

Wires the K=50 Weibull-stochastic AEP objective from stochastic_aep.py into a
copy of the FunWake SGD skeleton (playground/skeleton.py mirror with `key`
threaded through the carry), then runs multistart with both schedules on the
DEI training farm with iter_192's discovery wake model (Bastankhah k=0.04).

The deployed playground/skeleton.py is NOT touched. Matrix eval reproducibility
preserved.

Schedules sourced from paper_schedules/scripts/schedules.py (Claude iter_192
verbatim + pixwake sgd_baseline) and inlined here so this script is
self-contained.

Usage:
    PYTHONPATH=dependencies/pixwake/src pixi run python \\
        validation/stochastic_aep/run_part3.py \\
        --restarts 20 \\
        --out validation/stochastic_aep/part3_result.json
"""
import argparse
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pixwake.optim.sgd import boundary_penalty, spacing_penalty

from stochastic_aep import (
    build_sim,
    deterministic_fine_grid_aep,
    stochastic_aep_factory,
)


# --- Schedules (inlined from /Users/julianquick/clusters/paper_schedules/scripts/schedules.py) ---

def sgd_baseline(lr_init=50.0):
    """Pixwake baseline. Constant LR, no decay (matches FunWake iter_0 baseline).
    beta1=0.1, beta2=0.2."""
    def apply(step, total_steps, lr0, alpha0):
        lr = jnp.full((), float(lr_init), dtype=jnp.float64)
        alpha = alpha0
        beta1 = jnp.float64(0.1)
        beta2 = jnp.float64(0.2)
        return lr, alpha, beta1, beta2
    return apply


def funwake_iter192(lr_init=50.0):
    """Claude iter_192 verbatim: warmup + cosine + dual bumps + alpha dip.
    Source: funwake/runs/schedule_only_5hr/iter_192.py
    Constants beta1=0.3, beta2=0.5."""
    lr0_setting = float(lr_init)

    def apply(step, total_steps, lr0, alpha0):
        lr_ref = lr0_setting
        t = step / total_steps
        lr_peak = 4.0 * lr_ref
        lr_min = lr_peak / 10000.0
        warmup_end = 0.05
        warmup_lr = lr_peak * t / warmup_end
        cosine_t = (t - warmup_end) / (1.0 - warmup_end)
        cosine_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
        lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
        bump1 = 0.2 * lr_peak * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
        bump2 = 0.3 * lr_peak * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
        lr = lr_base + bump1 + bump2

        alpha_base = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha_extra = 3.0 * alpha0 * late ** 2
        dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
        alpha = (alpha_base + alpha_extra) * (1.0 - dip)

        beta1 = jnp.float64(0.3)
        beta2 = jnp.float64(0.5)
        return lr, alpha, beta1, beta2
    return apply


# --- Stochastic skeleton (mirror of playground/skeleton.py with key threaded) ---

def wind_aware_init(boundary, min_spacing, n_target, weights, wd, seed):
    """Faithful copy of playground/skeleton.py:43–91 wind-aware grid init."""
    boundary = jnp.array(boundary, dtype=jnp.float64)
    weights = jnp.array(weights, dtype=jnp.float64)
    wd = jnp.array(wd, dtype=jnp.float64)
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    angle = dominant + jnp.pi / 2
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T
    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, ny),
    )
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

    n_verts = boundary.shape[0]
    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    ix, iy = cand_x[inside], cand_y[inside]

    key = jax.random.PRNGKey(seed)
    if len(ix) >= n_target:
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        return ix[indices], iy[indices]
    k1, k2 = jax.random.split(key)
    return (
        jax.random.uniform(k1, (n_target,), minval=float(x_min), maxval=float(x_max)),
        jax.random.uniform(k2, (n_target,), minval=float(y_min), maxval=float(y_max)),
    )


def run_with_stochastic_schedule(
    schedule_fn, sim, aep_stoch_fn, K, n_target, boundary, min_spacing,
    weights, wd, total_steps=8000, init_seed=0, sample_seed=0,
):
    """Stochastic-objective mirror of playground/skeleton.py:run_with_schedule.

    schedule_fn       : (step, total_steps, lr0, alpha0) -> (lr, alpha, beta1, beta2)
    aep_stoch_fn      : (x, y, key, K) -> AEP estimate (GWh, signed: maximize AEP)
    K                 : MC draws per gradient call
    init_seed         : controls wind-aware-grid init shuffle
    sample_seed       : controls per-iter stochastic sampling
    """
    boundary = jnp.array(boundary, dtype=jnp.float64)
    weights = jnp.array(weights, dtype=jnp.float64)

    # Objective: negative-AEP (minimize)
    def neg_aep(x, y, key):
        return -aep_stoch_fn(x, y, key, K)

    def con_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    grad_obj = jax.grad(neg_aep, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x, y = wind_aware_init(boundary, min_spacing, n_target, weights, wd, init_seed)

    # lr0, alpha0 from problem scale — compute once using a deterministic-style
    # gradient call with K large enough to keep noise small. We use 5*K samples
    # of the stochastic gradient at the init layout as a stable scale estimate.
    k_init = jax.random.PRNGKey(sample_seed + 1_000_001)
    g_estimates_x = []
    g_estimates_y = []
    for _ in range(5):
        k_init, sub = jax.random.split(k_init)
        gx_, gy_ = grad_obj(x, y, sub)
        g_estimates_x.append(gx_)
        g_estimates_y.append(gy_)
    gox = jnp.mean(jnp.stack(g_estimates_x), axis=0)
    goy = jnp.mean(jnp.stack(g_estimates_y), axis=0)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

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
            jx = gox_i + alpha * gcx
            jy = goy_i + alpha * gcy

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


def check_feasibility(x, y, boundary, min_spacing):
    """Boundary + 2D spacing satisfied? Returns (boundary_violation_m,
    min_pair_dist_m, feasible_bool)."""
    boundary = jnp.array(boundary)
    bp = float(boundary_penalty(x, y, boundary))
    # spacing_penalty returns positive iff any pair < min_spacing
    sp = float(spacing_penalty(x, y, float(min_spacing)))
    # Independent min-pair-distance for clarity
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2 + jnp.eye(n) * 1e30
    min_d = float(jnp.sqrt(jnp.min(d2)))
    feas = bool(bp == 0.0) and bool(sp == 0.0)
    return {
        "boundary_violation": bp,
        "spacing_penalty": sp,
        "min_pair_dist_m": min_d,
        "feasible": feas,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="playground/problem.json")
    p.add_argument("--resource", default="validation/stochastic_aep/dei_weibull_12.json")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--wake-model", default="bastankhah_0.04")
    p.add_argument("--ws-step", type=float, default=0.1,
                   help="Deterministic-eval ws step (m/s) for final scoring (smaller = more accurate)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.problem) as f:
        problem = json.load(f)
    with open(args.resource) as f:
        resource = json.load(f)

    sim, D = build_sim(problem, wake_model=args.wake_model)
    aep_stoch_fn = stochastic_aep_factory(sim, resource)

    boundary = problem["boundary_vertices"]
    min_spacing = float(problem["min_spacing_m"])
    n_target = int(problem["n_target"])
    # The resource's sector centers + frequencies are passed to the wind-aware
    # init; this preserves the deployed init recipe.
    weights = resource["sector_probability"]
    wd = resource["sector_centers_deg"]

    schedules = {
        "sgd_baseline": sgd_baseline(),
        "funwake_iter192": funwake_iter192(),
    }

    per_restart = {name: [] for name in schedules}
    summary = {}

    t_global = time.time()
    for name, sched in schedules.items():
        print(f"\n=== schedule: {name} ===", flush=True)
        for r in range(args.restarts):
            t0 = time.time()
            x_opt, y_opt = run_with_stochastic_schedule(
                sched, sim, aep_stoch_fn, args.K, n_target, boundary, min_spacing,
                weights, wd, total_steps=args.total_steps,
                init_seed=r, sample_seed=r + 100_000,
            )
            elapsed = time.time() - t0
            feas = check_feasibility(x_opt, y_opt, boundary, min_spacing)
            # Deterministic final-AEP scoring (Weibull-marginalized fine grid)
            try:
                aep_det = deterministic_fine_grid_aep(
                    sim, x_opt, y_opt, resource,
                    ws_min=4.0, ws_max=25.0, ws_step=args.ws_step, wd_step=1.0,
                )
            except Exception as e:
                aep_det = None
                feas["det_aep_error"] = str(e)[:200]
            entry = {
                "restart": r,
                "elapsed_s": round(elapsed, 1),
                "aep_gwh_det_weibull": aep_det,
                **feas,
            }
            per_restart[name].append(entry)
            tag = "✓" if feas["feasible"] else "✗"
            aep_str = f"{aep_det:.2f}" if aep_det is not None else "ERR"
            print(
                f"[{name} r={r}] elapsed={elapsed:.0f}s {tag} AEP={aep_str} "
                f"min_pair_d={feas['min_pair_dist_m']:.0f}m "
                f"bp={feas['boundary_violation']:.2e} sp={feas['spacing_penalty']:.2e}",
                flush=True,
            )

        # Summary per schedule
        aeps_all = np.array([e["aep_gwh_det_weibull"] for e in per_restart[name]
                             if e["aep_gwh_det_weibull"] is not None])
        feasibles = np.array([e["feasible"] for e in per_restart[name]])
        aeps_feas = np.array([e["aep_gwh_det_weibull"] for e in per_restart[name]
                              if e["feasible"] and e["aep_gwh_det_weibull"] is not None])
        summary[name] = {
            "n_restarts": int(args.restarts),
            "feasibility_rate": float(feasibles.mean()),
            "n_feasible": int(feasibles.sum()),
            "best_aep_all": float(aeps_all.max()) if aeps_all.size else None,
            "best_aep_feasible": float(aeps_feas.max()) if aeps_feas.size else None,
            "mean_aep_feasible": float(aeps_feas.mean()) if aeps_feas.size else None,
            "std_aep_feasible": float(aeps_feas.std(ddof=1)) if aeps_feas.size > 1 else None,
            "mean_aep_all": float(aeps_all.mean()) if aeps_all.size else None,
            "std_aep_all": float(aeps_all.std(ddof=1)) if aeps_all.size > 1 else None,
        }

    # Paired comparison on same restart indices
    pair_diff = []
    for r in range(args.restarts):
        a = per_restart["funwake_iter192"][r]["aep_gwh_det_weibull"]
        b = per_restart["sgd_baseline"][r]["aep_gwh_det_weibull"]
        if (
            a is not None and b is not None
            and per_restart["funwake_iter192"][r]["feasible"]
            and per_restart["sgd_baseline"][r]["feasible"]
        ):
            pair_diff.append(a - b)
    pair_diff = np.array(pair_diff)
    paired_summary = None
    if pair_diff.size > 1:
        mean = float(pair_diff.mean())
        std = float(pair_diff.std(ddof=1))
        se = std / np.sqrt(pair_diff.size)
        t = mean / se if se > 0 else float("nan")
        paired_summary = {
            "n_pairs": int(pair_diff.size),
            "mean_diff_iter192_minus_baseline_gwh": mean,
            "std_diff_gwh": std,
            "se_diff_gwh": se,
            "t_stat": t,
        }

    out = {
        "config": {
            "problem": args.problem,
            "resource": args.resource,
            "K": args.K,
            "total_steps": args.total_steps,
            "restarts": args.restarts,
            "wake_model": args.wake_model,
            "n_target": n_target,
            "min_spacing_m": min_spacing,
        },
        "elapsed_s_total": round(time.time() - t_global, 1),
        "per_restart": per_restart,
        "summary_per_schedule": summary,
        "paired_diff": paired_summary,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if paired_summary is not None:
        print("\n=== PAIRED DIFF (iter_192 − baseline) ===")
        print(json.dumps(paired_summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
