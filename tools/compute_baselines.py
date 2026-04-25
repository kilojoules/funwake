#!/usr/bin/env python
"""Compute multi-start SGD baselines for DEI farm variants.

Runs topfarm_sgd_solve N times with random grid initializations,
keeps the best. Outputs JSON with AEP, feasibility, and timing.

Usage:
    # 500 multi-start baseline for all DEI variants
    python tools/compute_baselines.py --n-starts 500

    # Budget-matched baseline (as many starts as fit in 5 hours)
    python tools/compute_baselines.py --time-budget 18000

    # Specific farms and start count
    python tools/compute_baselines.py --problems results/problem_dei_n30.json \
        --n-starts 50
"""
import argparse
import glob
import json
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "playground", "pixwake", "src"))
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty


def load_problem(path):
    with open(path) as f:
        return json.load(f)


def build_sim(info):
    D = info["rotor_diameter"]
    hub_height = info.get("hub_height", 150.0)
    t = info["turbine"]
    turb = Turbine(
        rotor_diameter=D, hub_height=hub_height,
        power_curve=Curve(
            ws=jnp.array(t["power_curve_ws"], dtype=float),
            values=jnp.array(t["power_curve_kw"], dtype=float)),
        ct_curve=Curve(
            ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float),
            values=jnp.array(t["ct_curve_ct"], dtype=float)))
    return WakeSimulation(turb, BastankhahGaussianDeficit(k=0.04))


def grid_init(boundary, n_target, min_spacing, seed=0):
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

    key = jax.random.PRNGKey(seed)
    if len(inside_x) >= n_target:
        idx = jax.random.choice(key, len(inside_x), (n_target,), replace=False)
        return inside_x[idx], inside_y[idx]
    else:
        x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
        return x, y


def run_single_start(objective, boundary, n_target, min_spacing, seed,
                     settings=None):
    if settings is None:
        settings = SGDSettings(
            learning_rate=50.0,
            max_iter=4000,
            additional_constant_lr_iterations=2000,
            beta1=0.1, beta2=0.2,
        )

    init_x, init_y = grid_init(boundary, n_target, min_spacing, seed=seed)
    t0 = time.time()
    opt_x, opt_y = topfarm_sgd_solve(
        objective, init_x, init_y, boundary, min_spacing, settings)
    elapsed = time.time() - t0
    aep = float(-objective(opt_x, opt_y))

    bnd_pen = float(boundary_penalty(opt_x, opt_y, boundary))
    dx = opt_x[:, None] - opt_x[None, :]
    dy = opt_y[:, None] - opt_y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e10)
    min_dist = float(jnp.min(dist))
    feasible = (bnd_pen < 1e-3) and (min_dist >= min_spacing * 0.99)

    return aep, feasible, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="*", default=None,
                   help="Problem JSONs (default: all DEI variants + ROWP)")
    p.add_argument("--n-starts", type=int, default=None,
                   help="Fixed number of starts")
    p.add_argument("--time-budget", type=int, default=None,
                   help="Time budget in seconds (run as many starts as fit)")
    p.add_argument("--output", default="results/baselines_multistart.json")
    args = p.parse_args()

    if args.n_starts is None and args.time_budget is None:
        args.n_starts = 500

    project_root = os.path.join(os.path.dirname(__file__), "..")

    if args.problems is None:
        args.problems = sorted(glob.glob(
            os.path.join(project_root, "results", "problem_dei_n*.json")))
        rowp = os.path.join(project_root, "results", "problem_rowp.json")
        if os.path.exists(rowp):
            args.problems.append(rowp)

    all_results = {}

    for prob_path in args.problems:
        prob_name = os.path.basename(prob_path).replace(".json", "")
        info = load_problem(prob_path)
        n_target = info["n_target"]
        min_spacing = info["min_spacing_m"]
        boundary = jnp.array(info["boundary_vertices"])

        sim = build_sim(info)
        wd = jnp.array(info["wind_rose"]["directions_deg"])
        ws = jnp.array(info["wind_rose"]["speeds_ms"])
        weights = jnp.array(info["wind_rose"]["weights"])

        def objective(x, y):
            r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
            pw = r.power()[:, :len(x)]
            return -jnp.sum(pw * weights[:, None]) * 8760 / 1e6

        print("=== %s (n=%d) ===" % (prob_name, n_target))

        best_aep = -float("inf")
        best_feasible_aep = -float("inf")
        total_starts = 0
        total_time = 0
        feasible_count = 0

        start_iter = 0
        while True:
            if args.n_starts and total_starts >= args.n_starts:
                break
            if args.time_budget and total_time >= args.time_budget:
                break

            aep, feasible, elapsed = run_single_start(
                objective, boundary, n_target, min_spacing,
                seed=start_iter)
            total_starts += 1
            total_time += elapsed
            start_iter += 1

            if aep > best_aep:
                best_aep = aep
            if feasible:
                feasible_count += 1
                if aep > best_feasible_aep:
                    best_feasible_aep = aep

            if total_starts % 10 == 0 or total_starts <= 5:
                print("  start %d: AEP=%.1f feas=%s (best=%.1f, best_feas=%.1f, %.0fs total)" % (
                    total_starts, aep, feasible, best_aep, best_feasible_aep, total_time))

        result = {
            "problem": prob_name,
            "n_target": n_target,
            "n_starts": total_starts,
            "total_time_s": round(total_time, 1),
            "best_aep": round(best_aep, 2),
            "best_feasible_aep": round(best_feasible_aep, 2) if best_feasible_aep > -float("inf") else None,
            "feasible_rate": round(feasible_count / total_starts, 3),
        }
        all_results[prob_name] = result
        print("  DONE: %d starts, best=%.1f, best_feas=%.1f, feas_rate=%.1f%%, %.0fs" % (
            total_starts, best_aep,
            best_feasible_aep if best_feasible_aep > -float("inf") else 0,
            100 * feasible_count / total_starts, total_time))

    out_path = os.path.join(project_root, args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nWrote %s" % out_path)


if __name__ == "__main__":
    main()
