"""Task A+B+C combined runner — per-iter AEP + gradient norms + α + lr trace.

At each probe point (every `probe_every` iters):
  - deterministic full-rose AEP at current layout (matches existing per-iter)
  - ‖grad_obj‖ : norm of stochastic AEP gradient (1 MC sample, K=50 internally)
  - ‖grad_con‖ : norm of constraint gradient (deterministic)
  - alpha     : penalty coefficient from schedule_fn
  - lr        : learning rate
  - bp, sp    : boundary and spacing penalty values

ratio r = ‖grad_obj‖ / (alpha × ‖grad_con‖) is the Task A discriminator.

Schedules supported:
  - claude_iter192 (Claude's deployed)
  - decay_es_eta_<f>  (topfarm decay+ES at given ηT in metres; e.g. decay_es_eta_5.0)

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_gradient_balance.py \\
        --cell <path> --schedule claude_iter192 --es-mode off \\
        --sample-seeds 100000 200000 300000 \\
        --probe-every 200 --out validation/stochastic_aep/gb_<tag>.json
"""
import argparse
import json
import os
import sys
import time
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(cell_path, schedule_name, es_mode, sample_seed, init_seed,
            K, total_steps, probe_every):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies/pixwake/src"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "validation/stochastic_aep"))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from pixwake.optim.sgd import boundary_penalty, spacing_penalty
    from stochastic_aep import build_sim
    from run_part3 import wind_aware_init, funwake_iter192
    from run_step3_rowp import _translate_to_local, topfarm_default_decay
    from matrix_categorical_aep import (
        categorical_rose_aep_factory,
        deterministic_full_rose_aep,
    )

    with open(os.path.join(PROJECT_ROOT, cell_path)) as f:
        problem = json.load(f)
    sim, D = build_sim(problem, wake_model="bastankhah_0.04")
    aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])
    boundary_local, ox, oy = _translate_to_local(problem["boundary_vertices"])
    n_target = int(problem["n_target"])
    min_spacing = float(problem["min_spacing_m"])
    weights = problem["wind_rose"]["weights"]
    wd = problem["wind_rose"]["directions_deg"]

    if schedule_name == "claude_iter192":
        sched = funwake_iter192()
    elif schedule_name == "claude_iter192_no_bumps":
        from schedules_ablation import funwake_iter192_no_bumps
        sched = funwake_iter192_no_bumps()
    elif schedule_name == "claude_iter192_baseline_alpha":
        from schedules_ablation import funwake_iter192_baseline_alpha
        sched = funwake_iter192_baseline_alpha()
    elif schedule_name.startswith("decay_es_eta_"):
        eta = float(schedule_name.split("_")[-1])
        sched = topfarm_default_decay(lr_init=50.0, gamma_min_factor=eta / 50.0,
                                       total_steps=total_steps)
    else:
        raise ValueError(f"unknown schedule {schedule_name}")

    es_enabled = es_mode in ("on_lr_init", "on_runmax")
    use_runmax = es_mode == "on_runmax"

    x_init, y_init = wind_aware_init(boundary_local, min_spacing, n_target,
                                      weights, wd, init_seed)
    bnd_j = jnp.array(boundary_local)
    bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))
    sp_init = float(spacing_penalty(jnp.array(x_init), jnp.array(y_init), float(min_spacing)))

    # Build objectives
    def neg_aep(x, y, key):
        return -aep_stoch_fn(x, y, key, K)

    def con_penalty(x, y):
        return boundary_penalty(x, y, bnd_j) + spacing_penalty(x, y, min_spacing)

    grad_obj_fn = jax.grad(neg_aep, argnums=(0, 1))
    grad_con_fn = jax.grad(con_penalty, argnums=(0, 1))

    # lr0 / alpha0 (matches run_step3 protocol)
    k_init = jax.random.PRNGKey(sample_seed + 1_000_001)
    g_x, g_y = [], []
    for _ in range(5):
        k_init, sub = jax.random.split(k_init)
        gx_, gy_ = grad_obj_fn(x_init, y_init, sub)
        g_x.append(gx_); g_y.append(gy_)
    gox = jnp.mean(jnp.stack(g_x), axis=0)
    goy = jnp.mean(jnp.stack(g_y), axis=0)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    # Precompute lr trajectory + ES first cross
    lr_traj = []
    alpha_traj = []
    for i in range(0, total_steps, probe_every):
        lr_v, a_v, _, _ = sched(int(i), total_steps, lr0, alpha0)
        lr_traj.append(float(lr_v))
        alpha_traj.append(float(a_v))
    lr_traj = np.asarray(lr_traj)
    alpha_traj = np.asarray(alpha_traj)
    if use_runmax:
        lr_ratio_traj = lr_traj / np.maximum(np.maximum.accumulate(lr_traj), 1e-10)
    else:
        lr_ratio_traj = lr_traj / 50.0
    crosses = lr_ratio_traj <= 0.1
    es_first_cross_iter = (
        int(np.where(crosses)[0][0]) * probe_every if crosses.any() else None
    )
    es_all_cross_iters = [int(i * probe_every) for i in np.where(crosses)[0].tolist()]

    @jax.jit
    def run_chunk(x, y, mx, my, vx, vy, key, max_lr_so_far, start_step):
        def step(j, carry):
            x, y, mx, my, vx, vy, key, max_lr_so_far = carry
            i = start_step + j
            key, subkey = jax.random.split(key)
            lr, alpha, b1, b2 = sched(i, total_steps, lr0, alpha0)
            new_max_lr = jnp.maximum(max_lr_so_far, lr)
            gox_i, goy_i = grad_obj_fn(x, y, subkey)
            gcx, gcy = grad_con_fn(x, y)
            if es_enabled:
                if use_runmax:
                    lr_ratio = lr / jnp.maximum(new_max_lr, 1e-10)
                else:
                    lr_ratio = lr / lr0
                es_active = lr_ratio <= 0.1
                zero_x = jnp.zeros_like(gox_i); zero_y = jnp.zeros_like(goy_i)
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
            eps = 1e-12
            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)
            return (x_new, y_new, mx_new, my_new, vx_new, vy_new, key, new_max_lr)
        init = (x, y, mx, my, vx, vy, key, max_lr_so_far)
        return jax.lax.fori_loop(0, probe_every, step, init)

    x, y = x_init, y_init
    mx = jnp.zeros_like(x); my = jnp.zeros_like(y)
    vx = jnp.zeros_like(x); vy = jnp.zeros_like(y)
    key0 = jax.random.PRNGKey(sample_seed)
    max_lr0 = jnp.array(1e-10, dtype=jnp.float64)

    def probe(x, y, iter_i):
        # AEP
        aep = float(deterministic_full_rose_aep(sim, x, y, problem["wind_rose"]))
        # Constraint penalties at this layout
        bp = float(boundary_penalty(x, y, bnd_j))
        sp = float(spacing_penalty(x, y, min_spacing))
        # Gradient norms (use a probe-specific key, deterministic given seed)
        key_probe = jax.random.PRNGKey(sample_seed + 2_000_000 + iter_i)
        gox_p, goy_p = grad_obj_fn(x, y, key_probe)
        gcx_p, gcy_p = grad_con_fn(x, y)
        gobj_norm = float(jnp.sqrt(jnp.sum(gox_p**2) + jnp.sum(goy_p**2)))
        gcon_norm = float(jnp.sqrt(jnp.sum(gcx_p**2) + jnp.sum(gcy_p**2)))
        lr_v, alpha_v, _, _ = sched(int(iter_i), total_steps, lr0, alpha0)
        alpha_f = float(alpha_v); lr_f = float(lr_v)
        alpha_gcon = alpha_f * gcon_norm
        ratio = gobj_norm / (alpha_gcon + 1e-30)
        return {
            "iter": int(iter_i),
            "aep": aep, "bp": bp, "sp": sp,
            "grad_obj_norm": gobj_norm,
            "grad_con_norm": gcon_norm,
            "alpha": alpha_f, "lr": lr_f,
            "alpha_grad_con_norm": alpha_gcon,
            "ratio_obj_over_alpha_con": ratio,
        }

    # Initial probe
    trace = [probe(x, y, 0)]
    t0 = time.time()
    n_chunks = total_steps // probe_every
    for c in range(n_chunks):
        x, y, mx, my, vx, vy, key0, max_lr0 = run_chunk(
            x, y, mx, my, vx, vy, key0, max_lr0, c * probe_every,
        )
        it = (c + 1) * probe_every
        trace.append(probe(x, y, it))
    elapsed = time.time() - t0

    return {
        "cell_path": cell_path,
        "schedule": schedule_name,
        "es_mode": es_mode,
        "sample_seed": sample_seed,
        "init_seed": init_seed,
        "K": K,
        "total_steps": total_steps,
        "probe_every": probe_every,
        "trace": trace,
        "es_first_cross_iter": es_first_cross_iter,
        "es_all_cross_iters": es_all_cross_iters,
        "final_aep_gwh": trace[-1]["aep"],
        "final_bp": trace[-1]["bp"],
        "final_sp": trace[-1]["sp"],
        "elapsed_s": round(elapsed, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--es-mode", choices=["off", "on_lr_init", "on_runmax"], default="off")
    p.add_argument("--sample-seeds", type=int, nargs="+", default=[100000, 200000, 300000])
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"config": vars(args), "runs": []}
    seen = {r["sample_seed"] for r in results["runs"]}

    for ss in args.sample_seeds:
        if ss in seen:
            continue
        print(f"=== {args.cell} {args.schedule} es={args.es_mode} ss={ss} ===", flush=True)
        try:
            r = run_one(args.cell, args.schedule, args.es_mode, ss, args.init_seed,
                         args.K, args.total_steps, args.probe_every)
        except Exception as e:
            r = {"sample_seed": ss, "error": str(e)[:300], "trace_err": traceback.format_exc()[:500]}
        results["runs"].append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if "error" not in r:
            print(f"  final AEP={r['final_aep_gwh']:.2f} bp={r['final_bp']:.1e} "
                   f"es_first_cross={r['es_first_cross_iter']} elapsed={r['elapsed_s']}s",
                   flush=True)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
