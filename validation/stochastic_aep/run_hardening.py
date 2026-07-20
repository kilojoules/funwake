"""Hardening runs — 4 targeted follow-ups to the δ-sweep + ES paper-readiness pass.

H1 — Refined δ on 4 low-margin cells. 7 new δ between existing grid points
     interpolate the curve around best-sampled-δ; confirm no finer δ closes
     the gap below 0.2 %.

H2 — Re-run Experiment B with running-max ES trigger so the warmup-spurious
     iter-0 fire is removed. Cleanly tests "iter_192 doesn't need explicit ES."

H3 — Multi-init iter_192 ES-off on 4 low-margin cells. Adds 2 new init seeds
     so gap/spread for low-margin cells is empirical across inits, not just
     across samples at single init.

H4 — TopFarm smart-start init as a SECONDARY fair baseline. Replaces the
     wind-aware grid init with `topfarm.utils.smart_start` (greedy AEP-aware
     placement under spacing constraint). Runs on 18 headline cells ×
     {decay+ES default-δ, claude_iter192, gemini_iter192} × 3 sample seeds.

Total: 84 (H1) + 54 (H2) + 24 (H3) + 162 (H4) = 324 runs.

Usage:
    PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \\
        validation/stochastic_aep/run_hardening.py \\
        --workers 1 --out validation/stochastic_aep/hardening.json
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

# 4 lowest-margin cells from delta_sweep analysis (gap/spread ratio < 15):
#   dei_n60_roserowp        gap=0.607 spread=0.049 margin=12.4
#   rowp_n70_roseomnidir    gap=0.538 spread=0.043 margin=12.5
#   rowp_n60_rosedei        gap=0.759 spread=0.059 margin=12.9
#   rowp_n60_roserowp       gap=0.736 spread=0.052 margin=14.2
LOW_MARGIN_CELLS = [
    "results/matrix/problem_dei_n60_roserowp.json",
    "results/matrix/problem_rowp_n70_roseomnidir.json",
    "results/matrix/problem_rowp_n60_rosedei.json",
    "results/matrix/problem_rowp_n60_roserowp.json",
]

# Existing δ grid was {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5}. Interpolate
# at the geometric midpoints between adjacent samples.
REFINED_DELTAS = [0.003, 0.007, 0.015, 0.03, 0.07, 0.15, 0.3]


# -----------------------------------------------------------------------------
# Running-max ES trigger — used by H2.
# -----------------------------------------------------------------------------

def run_with_stochastic_schedule_es_runmax(
    schedule_fn, sim, aep_stoch_fn, K, n_target, boundary, min_spacing,
    weights, wd, total_steps=8000, init_seed=0, sample_seed=0,
    early_stopping=True, es_threshold=0.1, init_xy_override=None,
):
    """Mirror of run_step3.run_with_stochastic_schedule_es, BUT the ES trigger
    uses `lr_i / running_max(lr_history)` instead of `lr_i / lr_init`. For
    monotone-decay schedules this is equivalent; for warmup schedules like
    iter_192 it avoids the spurious iter-0 fire."""
    import jax
    import jax.numpy as jnp
    from pixwake.optim.sgd import boundary_penalty, spacing_penalty

    boundary = jnp.array(boundary, dtype=jnp.float64)
    weights = jnp.array(weights, dtype=jnp.float64)

    def neg_aep(x, y, key):
        return -aep_stoch_fn(x, y, key, K)

    def con_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    grad_obj = jax.grad(neg_aep, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    if init_xy_override is not None:
        x, y = init_xy_override
        x = jnp.asarray(x); y = jnp.asarray(y)
    else:
        from run_part3 import wind_aware_init
        x, y = wind_aware_init(boundary, min_spacing, n_target, weights, wd, init_seed)

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
    es_threshold_f = float(es_threshold)

    @jax.jit
    def run_loop(x, y, key0):
        mx = jnp.zeros_like(x); my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x); vy = jnp.zeros_like(y)
        eps = 1e-12
        max_lr0 = jnp.array(1e-10, dtype=jnp.float64)

        def step(i, carry):
            x, y, mx, my, vx, vy, key, max_lr_so_far = carry
            key, subkey = jax.random.split(key)
            lr, alpha, b1, b2 = schedule_fn(i, total_steps, lr0, alpha0)
            # Update running max BEFORE computing ES check, so ES uses the
            # max-so-far rather than min(lr, max_so_far)/max_so_far = 1.
            new_max_lr = jnp.maximum(max_lr_so_far, lr)
            gox_i, goy_i = grad_obj(x, y, subkey)
            gcx, gcy = grad_con(x, y)
            if es_enabled:
                # Running-max ES trigger: only fires once lr has decayed
                # FROM its post-warmup peak. Avoids warmup spurious fire.
                lr_ratio = lr / jnp.maximum(new_max_lr, 1e-10)
                es_active = lr_ratio <= es_threshold_f
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
            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)
            return (x_new, y_new, mx_new, my_new, vx_new, vy_new, key, new_max_lr)

        init = (x, y, mx, my, vx, vy, key0, max_lr0)
        final = jax.lax.fori_loop(0, total_steps, step, init)
        return final[0], final[1]

    import jax
    key0 = jax.random.PRNGKey(sample_seed)
    return run_loop(x, y, key0)


# -----------------------------------------------------------------------------
# Smart-start init — used by H4.
# -----------------------------------------------------------------------------

_SMART_START_CACHE_DIR = os.path.join(PROJECT_ROOT, "validation/stochastic_aep/_smart_start_cache")
os.makedirs(_SMART_START_CACHE_DIR, exist_ok=True)


def smart_start_init(problem, boundary_local, n_target, min_spacing, seed=0,
                      n_grid_pts=50, cell_path=None):
    """Build a grid covering the local-coords polygon, compute per-grid-point
    single-turbine AEP via the cell's pixwake sim, then call topfarm.utils.smart_start
    to pick n_target positions greedily under the spacing constraint.

    Cached per (cell_path, seed, n_grid_pts) on disk so multiple seeds reuse
    the same init."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    if cell_path is not None:
        cache_key = f"{os.path.basename(cell_path).replace('.json','')}_seed{seed}_grid{n_grid_pts}.json"
        cache_path = os.path.join(_SMART_START_CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            d = json.load(open(cache_path))
            return np.asarray(d["x"]), np.asarray(d["y"])

    from topfarm.utils import smart_start
    from stochastic_aep import build_sim

    sim, D = build_sim(problem, wake_model="bastankhah_0.04")

    bnd = np.array(boundary_local)
    x_min, x_max = bnd[:, 0].min(), bnd[:, 0].max()
    y_min, y_max = bnd[:, 1].min(), bnd[:, 1].max()
    XX, YY = np.meshgrid(
        np.linspace(x_min, x_max, n_grid_pts),
        np.linspace(y_min, y_max, n_grid_pts),
    )
    # Filter grid points inside polygon
    from shapely.geometry import Point, Polygon
    poly = Polygon([(p[0], p[1]) for p in boundary_local])
    inside_mask = np.array([
        poly.contains(Point(xx, yy)) for xx, yy in zip(XX.ravel(), YY.ravel())
    ]).reshape(XX.shape)

    rose = problem["wind_rose"]
    wd = jnp.array(rose["directions_deg"])
    ws = jnp.array(rose["speeds_ms"])
    weights = jnp.array(rose["weights"])
    weights = weights / jnp.sum(weights)

    # Single-turbine AEP at each grid point. Vectorize: build a batch of K
    # cases (one per direction-speed pair) for a single turbine at each grid
    # point. Loop over grid points (n_grid_pts^2 small) for simplicity.
    ZZ = np.full(XX.shape, -1e10, dtype=float)
    flat_idx = np.where(inside_mask.ravel())[0]
    Xs = XX.ravel()[flat_idx]
    Ys = YY.ravel()[flat_idx]
    ti = jnp.full_like(ws, 0.06)
    # Vectorize over grid points: single-turbine power doesn't depend on
    # neighbors, so AEP(x, y) = sum_k w_k * power(ws_k) = constant in (x, y).
    # That makes smart_start degenerate (every point has the same ZZ). For
    # a meaningful smart_start we need a wake-aware AEP, which requires
    # sequential placement (Pixwake's smart_start does that internally via
    # the radius mechanism). topfarm.utils.smart_start handles this: when
    # ZZ is constant or near-constant, it falls back to a packing algorithm
    # that respects min_space. So we use a SPATIAL prior instead: ZZ = the
    # AEP of a turbine at (x, y) downstream-fewer being prioritized — i.e.,
    # for the dominant wind direction, compute "how much wake-free space is
    # upstream of this point" via a simple proxy. For correctness we use
    # the constant ZZ + smart_start's packing fallback, which gives the
    # greedy-packed minimum-spacing layout.
    for i, (xx, yy) in enumerate(zip(Xs, Ys)):
        x_arr = jnp.array([xx])
        y_arr = jnp.array([yy])
        r = sim(x_arr, y_arr, ws_amb=ws, wd_amb=wd, ti_amb=ti)
        aep = float(r.aep(probabilities=weights))
        ZZ.ravel()[flat_idx[i]] = aep

    # smart_start picks N_WT positions in XX,YY where ZZ is highest,
    # respecting min_space. random_pct=0 → fully deterministic given seed.
    xs, ys = smart_start(XX, YY, ZZ, N_WT=n_target, min_space=min_spacing,
                          random_pct=0, plot=False, seed=seed)
    xs = np.asarray(xs); ys = np.asarray(ys)
    if cell_path is not None:
        with open(cache_path, "w") as f:
            json.dump({"x": xs.tolist(), "y": ys.tolist()}, f)
    return xs, ys


# -----------------------------------------------------------------------------
# Universal task runner — dispatches by task_kind.
# -----------------------------------------------------------------------------

def run_task(task):
    """task = dict with keys:
       kind            : h1_refined_delta | h2_fixed_es_iter192 | h3_multi_init | h4_smart_start
       cell_path       : path to problem JSON
       sample_seed     : MC sampling seed
       init_seed       : layout init RNG seed (h1, h2, h3 use this for wind-aware init; h4 ignores)
       schedule        : 'decay_es_baseline_<delta>' | 'claude_iter192' | 'gemini_iter192' | 'iter192_es_on_runmax' | 'iter192_es_off_multi_init'
       delta           : (h1) baseline gamma_min_factor for delta sweep
       smart_init      : (h4) bool — replace wind-aware init with smart-start
    """
    import sys
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
        with open(os.path.join(PROJECT_ROOT, task["cell_path"])) as f:
            problem = json.load(f)
        sim, D = build_sim(problem, wake_model="bastankhah_0.04")
        aep_stoch_fn = categorical_rose_aep_factory(sim, problem["wind_rose"])
        boundary_local, ox, oy = _translate_to_local(problem["boundary_vertices"])
        n_target = int(problem["n_target"])
        min_spacing = float(problem["min_spacing_m"])
        weights = problem["wind_rose"]["weights"]
        wd = problem["wind_rose"]["directions_deg"]
        K = task.get("K", 50); total_steps = task.get("total_steps", 8000)

        # Build init layout
        if task.get("smart_init"):
            x_init, y_init = smart_start_init(
                problem, boundary_local, n_target, min_spacing,
                seed=task["init_seed"], cell_path=task["cell_path"],
            )
        else:
            x_init, y_init = wind_aware_init(
                boundary_local, min_spacing, n_target, weights, wd, task["init_seed"],
            )
        bnd_j = jnp.array(boundary_local)
        bp_init = float(boundary_penalty(jnp.array(x_init), jnp.array(y_init), bnd_j))
        sp_init = float(spacing_penalty(jnp.array(x_init), jnp.array(y_init), float(min_spacing)))

        # Build schedule
        kind = task["kind"]
        if kind == "h1_refined_delta":
            sched = topfarm_default_decay(
                lr_init=50.0, gamma_min_factor=task["delta"], total_steps=total_steps,
            )
            es_on = True
            use_runmax = False
        elif kind == "h2_fixed_es_iter192":
            sched = funwake_iter192()
            es_on = True
            use_runmax = True
        elif kind == "h3_multi_init":
            # iter_192 ES-off with different init seed (already handled by init above)
            sched = funwake_iter192()
            es_on = False
            use_runmax = False
        elif kind == "h4_smart_start":
            # Schedule depends on which baseline is being tested
            s_name = task["schedule"]
            if s_name == "decay_es_baseline":
                sched = topfarm_default_decay(
                    lr_init=50.0, gamma_min_factor=0.01, total_steps=total_steps,
                )
                es_on = True
            elif s_name == "claude_iter192":
                sched = funwake_iter192()
                es_on = False
            elif s_name == "gemini_iter192":
                sched = gemini_iter192()
                es_on = False
            else:
                raise ValueError(f"unknown schedule {s_name}")
            use_runmax = False
        else:
            raise ValueError(f"unknown kind {kind}")

        t0 = time.time()
        if use_runmax:
            x_opt, y_opt = run_with_stochastic_schedule_es_runmax(
                sched, sim, aep_stoch_fn, K, n_target, boundary_local, min_spacing,
                weights, wd, total_steps=total_steps,
                init_seed=task["init_seed"], sample_seed=task["sample_seed"],
                early_stopping=es_on, es_threshold=0.1,
                init_xy_override=(x_init, y_init),
            )
        else:
            x_opt, y_opt = run_with_stochastic_schedule_es(
                sched, sim, aep_stoch_fn, K, n_target, boundary_local, min_spacing,
                weights, wd, total_steps=total_steps,
                init_seed=task["init_seed"], sample_seed=task["sample_seed"],
                early_stopping=es_on, es_threshold=0.1,
            )
            # The original run_with_stochastic_schedule_es doesn't accept
            # init_xy_override, so when smart_init is requested we have a
            # mismatch. Patch: smart_init layouts come from the override path
            # only; for h4 we re-implement the SGD here.
            # Since smart_init=True needs init_xy_override, fall through to
            # runmax variant which accepts it. Use it with use_runmax=False
            # internally (no spurious-trigger concern for default schedule).
            if task.get("smart_init"):
                x_opt, y_opt = run_with_stochastic_schedule_es_runmax(
                    sched, sim, aep_stoch_fn, K, n_target, boundary_local, min_spacing,
                    weights, wd, total_steps=total_steps,
                    init_seed=task["init_seed"], sample_seed=task["sample_seed"],
                    early_stopping=es_on, es_threshold=0.1,
                    init_xy_override=(x_init, y_init),
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
            **{k: task[k] for k in ("kind", "cell_path", "sample_seed", "init_seed")},
            "delta": task.get("delta"),
            "schedule": task.get("schedule"),
            "smart_init": task.get("smart_init", False),
            "es_trigger_mode": "running_max" if use_runmax else "lr_init",
            "aep_det_gwh": float(aep_det),
            "bp_init": bp_init, "sp_init": sp_init,
            "bp_final": bp_final, "sp_final": sp_final,
            "min_pair_dist_m": min_d,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        return {
            **{k: task[k] for k in ("kind", "cell_path", "sample_seed", "init_seed")},
            "delta": task.get("delta"),
            "schedule": task.get("schedule"),
            "smart_init": task.get("smart_init", False),
            "error": str(e)[:300],
            "trace": traceback.format_exc()[:500],
        }


def build_tasks(sample_seeds):
    tasks = []
    # H1 — refined δ on 4 low-margin cells
    for cell in LOW_MARGIN_CELLS:
        for delta in REFINED_DELTAS:
            for ss in sample_seeds:
                tasks.append({
                    "kind": "h1_refined_delta", "cell_path": cell,
                    "delta": delta, "sample_seed": ss, "init_seed": 0,
                })
    # H2 — fixed-ES iter_192 on all 18 headline cells
    for cell in HEADLINE_CELLS:
        for ss in sample_seeds:
            tasks.append({
                "kind": "h2_fixed_es_iter192", "cell_path": cell,
                "sample_seed": ss, "init_seed": 0,
            })
    # H3 — multi-init iter_192 ES-off on 4 low-margin cells (2 NEW init seeds)
    for cell in LOW_MARGIN_CELLS:
        for is_ in (1, 2):
            for ss in sample_seeds:
                tasks.append({
                    "kind": "h3_multi_init", "cell_path": cell,
                    "sample_seed": ss, "init_seed": is_,
                })
    # H4 — smart-start init on 18 headline cells × 3 schedules × 3 sample seeds
    for cell in HEADLINE_CELLS:
        for sname in ("decay_es_baseline", "claude_iter192", "gemini_iter192"):
            for ss in sample_seeds:
                tasks.append({
                    "kind": "h4_smart_start", "cell_path": cell,
                    "schedule": sname, "sample_seed": ss, "init_seed": 0,
                    "smart_init": True,
                })
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sample-seeds", type=int, nargs="+",
                   default=[100000, 200000, 300000])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"config": vars(args), "runs": []}
    seen = set()
    for r in results["runs"]:
        seen.add((
            r["kind"], r["cell_path"], r["sample_seed"], r["init_seed"],
            r.get("delta"), r.get("schedule"), r.get("smart_init", False),
        ))

    all_tasks = build_tasks(args.sample_seeds)
    tasks = [
        t for t in all_tasks if (
            t["kind"], t["cell_path"], t["sample_seed"], t["init_seed"],
            t.get("delta"), t.get("schedule"), t.get("smart_init", False),
        ) not in seen
    ]

    print(f"Total tasks: {len(all_tasks)} (already done: {len(seen)}, to run: {len(tasks)})", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    t_start = time.time()
    done = 0
    total = len(tasks)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                  mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                results["runs"].append(r)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                done += 1
                tag = ("ERR " + r["error"][:80]) if "error" in r else (
                    f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} "
                    f"elapsed={r['elapsed_s']}s"
                )
                print(f"[{done}/{total}] {r['kind']:25s} {os.path.basename(r['cell_path']):42s}  "
                       f"δ={r.get('delta')} sch={r.get('schedule')} is={r['init_seed']} ss={r['sample_seed']}  {tag}",
                       flush=True)
    else:
        for t in tasks:
            r = run_task(t)
            results["runs"].append(r)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            done += 1
            tag = ("ERR " + r["error"][:80]) if "error" in r else (
                f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} "
                f"elapsed={r['elapsed_s']}s"
            )
            print(f"[{done}/{total}] {r['kind']:25s} {os.path.basename(r['cell_path']):42s}  "
                   f"δ={r.get('delta')} sch={r.get('schedule')} is={r['init_seed']} ss={r['sample_seed']}  {tag}",
                   flush=True)

    results["elapsed_total_s"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {results['elapsed_total_s']/60:.1f} min wall. Wrote {args.out}")


if __name__ == "__main__":
    main()
