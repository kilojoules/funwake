#!/usr/bin/env python
"""Run a single multi-start baseline and print JSON result.

Usage:
    python tools/run_single_baseline.py results/problem_dei_n50.json --seed 42
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "playground", "pixwake", "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", help="Path to problem JSON")
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    info = json.load(open(args.problem))
    D = info["rotor_diameter"]
    t = info["turbine"]
    turb = Turbine(
        rotor_diameter=D, hub_height=info.get("hub_height", 150.0),
        power_curve=Curve(ws=jnp.array(t["power_curve_ws"], dtype=float),
                          values=jnp.array(t["power_curve_kw"], dtype=float)),
        ct_curve=Curve(ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float),
                       values=jnp.array(t["ct_curve_ct"], dtype=float)))
    sim = WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))

    wd = jnp.array(info["wind_rose"]["directions_deg"])
    ws = jnp.array(info["wind_rose"]["speeds_ms"])
    weights = jnp.array(info["wind_rose"]["weights"])
    boundary = jnp.array(info["boundary_vertices"])
    n_target = info["n_target"]
    min_spacing = info["min_spacing_m"]

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        pw = r.power()[:, :len(x)]
        return -jnp.sum(pw * weights[:, None]) * 8760 / 1e6

    # Grid init with random subsampling
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    nx = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny))
    cand_x, cand_y = gx.flatten(), gy.flatten()
    n_verts = boundary.shape[0]
    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    inside_x, inside_y = cand_x[inside], cand_y[inside]

    key = jax.random.PRNGKey(args.seed)
    if len(inside_x) >= n_target:
        idx = jax.random.choice(key, len(inside_x), (n_target,), replace=False)
        init_x, init_y = inside_x[idx], inside_y[idx]
    else:
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    settings = SGDSettings(learning_rate=50.0, max_iter=4000,
                           additional_constant_lr_iterations=2000,
                           beta1=0.1, beta2=0.2)

    t0 = time.time()
    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                      boundary, min_spacing, settings)
    elapsed = time.time() - t0

    aep = float(-objective(opt_x, opt_y))
    bnd_pen = float(boundary_penalty(opt_x, opt_y, boundary))
    dx = opt_x[:, None] - opt_x[None, :]
    dy = opt_y[:, None] - opt_y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e10)
    min_dist = float(jnp.min(dist))
    feasible = (bnd_pen < 1e-3) and (min_dist >= min_spacing * 0.99)

    print(json.dumps({
        "seed": args.seed,
        "aep": round(aep, 2),
        "feasible": feasible,
        "time": round(elapsed, 1),
    }))


if __name__ == "__main__":
    main()
