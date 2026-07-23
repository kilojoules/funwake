#!/usr/bin/env python
"""Re-evaluate a schedule-only file with a PARAMETERIZED lr0.

Replicates the tools/run_optimizer.py --schedule-only pipeline
(playground/harness.py + playground/skeleton.py) exactly, except that
lr0 — hardcoded to 50.0 at playground/skeleton.py:102 — is a parameter.
The coupling alpha0 = mean|grad_obj| / lr0 (skeleton.py:103) is kept, so
changing lr0 also changes alpha0, exactly as "the skeleton with a
different lr0" would.

Scoring uses benchmarks/dei_layout.ProblemBenchmark (same as
run_optimizer._score_layout): feasible = spacing_ok AND boundary_ok,
where boundary_ok = boundary_penalty < 1e-3 and
spacing_ok = min pairwise distance >= 0.99 * min_spacing.

Usage:
    pixi run python tools/reeval_lr0.py <schedule.py> \
        --problem results/problem_dei_n50.json \
        --lr0 240            # a float, or "rd" -> problem rotor_diameter \
        --out results/lr0_rd_reeval/train/iter_001.json

Writes a JSON result to --out (and prints it). On failure writes
{"error": ...} so drivers can skip-and-continue.
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dependencies", "pixwake", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks"))

import jax
jax.config.update("jax_enable_x64", True)

import importlib.util
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def build_sim(info):
    """Identical to playground/harness.py build_sim."""
    D = info["rotor_diameter"]
    hub_height = info.get("hub_height", 150.0)
    t = info["turbine"]
    ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
    power = jnp.array(t["power_curve_kw"], dtype=float)
    ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
    ct = jnp.array(t["ct_curve_ct"], dtype=float)
    turbine = Turbine(
        rotor_diameter=D, hub_height=hub_height,
        power_curve=Curve(ws=ws_arr, values=power),
        ct_curve=Curve(ws=ct_ws, values=ct))
    return WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))


def run_with_schedule(schedule_fn, sim, n_target, boundary, min_spacing,
                      wd, ws, weights, lr0_value, total_steps=8000, seed=0):
    """Copy of playground/skeleton.py run_with_schedule (early_stopping off,
    the harness default), with lr0 parameterized instead of hardcoded 50.0.
    """
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def con_penalty(x, y):
        return (boundary_penalty(x, y, boundary)
                + spacing_penalty(x, y, min_spacing))

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    # ── Wind-aware grid initialization (identical to skeleton) ─────
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
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
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
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

    if len(ix) >= n_target:
        key = jax.random.PRNGKey(seed)
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        x, y = ix[indices], iy[indices]
    else:
        key = jax.random.PRNGKey(seed)
        x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── lr0 / alpha0: the ONLY deviation from the skeleton ─────────
    gox, goy = grad_obj(x, y)
    lr0 = float(lr0_value)                                   # skeleton: 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0  # coupled

    @jax.jit
    def run_loop(x, y):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        eps = 1e-12

        def step(i, carry):
            x, y, mx, my, vx, vy = carry
            lr, alpha, b1, b2 = schedule_fn(i, total_steps, lr0, alpha0)
            gox, goy = grad_obj(x, y)
            gcx, gcy = grad_con(x, y)
            jx = gox + alpha * gcx
            jy = goy + alpha * gcy
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
            return (x_new, y_new, mx_new, my_new, vx_new, vy_new)

        init = (x, y, mx, my, vx, vy)
        final = jax.lax.fori_loop(0, total_steps, step, init)
        return final[0], final[1]

    return run_loop(x, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Path to schedule module (defines schedule_fn)")
    p.add_argument("--problem", required=True)
    p.add_argument("--lr0", required=True,
                   help='Float, or "rd" to use the problem rotor_diameter')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--out", default=None, help="Path to write result JSON")
    args = p.parse_args()

    result = {
        "file": os.path.basename(args.script),
        "problem": os.path.basename(args.problem),
        "seed": args.seed,
    }
    t0 = time.time()
    try:
        with open(args.problem) as f:
            info = json.load(f)

        lr0_value = (float(info["rotor_diameter"]) if args.lr0 == "rd"
                     else float(args.lr0))
        result["lr0"] = lr0_value

        spec = importlib.util.spec_from_file_location("schedule_mod", args.script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "schedule_fn"):
            raise RuntimeError("module does not define schedule_fn")
        if hasattr(mod, "optimize"):
            raise RuntimeError("module defines optimize() — rejected in "
                               "schedule-only mode")

        sim = build_sim(info)
        wd = jnp.array(info["wind_rose"]["directions_deg"])
        ws = jnp.array(info["wind_rose"]["speeds_ms"])
        weights = jnp.array(info["wind_rose"]["weights"])
        boundary = jnp.array(info["boundary_vertices"])

        opt_x, opt_y = run_with_schedule(
            mod.schedule_fn, sim, info["n_target"], boundary,
            info["min_spacing_m"], wd, ws, weights,
            lr0_value, total_steps=args.steps, seed=args.seed)
        xs = [float(v) for v in opt_x]
        ys = [float(v) for v in opt_y]

        from dei_layout import ProblemBenchmark
        bm = ProblemBenchmark(os.path.abspath(args.problem))
        aep = bm.score(xs, ys)
        feas = bm.check_feasibility(xs, ys)

        result.update({
            "aep_gwh": round(float(aep), 2),
            "feasible": bool(feas["spacing_ok"] and feas["boundary_ok"]),
            "boundary_ok": bool(feas["boundary_ok"]),
            "spacing_ok": bool(feas["spacing_ok"]),
            "boundary_penalty": float(feas["boundary_penalty"]),
            "min_dist": round(float(feas["min_turbine_distance_m"]), 2),
            "time_s": round(time.time() - t0, 1),
        })
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"[:500]
        result["time_s"] = round(time.time() - t0, 1)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
