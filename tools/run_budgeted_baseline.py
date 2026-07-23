#!/usr/bin/env python
"""Equivalent-cost budgeted SGD baseline harness.

Gives topfarm-style SGD a fixed budget of gradient evaluations (default
8000, matching one schedule-mode run), spent on K vmapped multistarts of
T = budget // K iterations each, with Quick-2023 early stopping and a
decay schedule scaled to fit T. Reports the best feasible layout's AEP.

Usage:
    pixi run python tools/run_budgeted_baseline.py results/problem_dei_n50.json \
        --starts 4 [--budget 8000] [--iters T] [--seed 0] [--no-es] \
        [--out out.json] [--save-layout layout.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dependencies", "pixwake", "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (
    SGDSettings,
    _compute_mid_bisection,
    boundary_penalty,
    topfarm_sgd_solve_multistart,
)

# --- Schedule scaling constants (recalibratable) -----------------------------
# Given T iterations per start:
#   constant-lr phase length C = round(CONST_FRAC * T)
#   decay design length    M = round(DECAY_FRAC * T)
# The decay shape is an M-step schedule (lr0 -> GAMMA_MIN) truncated at T.
CONST_FRAC = 0.45
DECAY_FRAC = 0.90

LR0 = 50.0
GAMMA_MIN = 0.01          # absolute final lr of the M-step decay design
ES_THRESHOLD = 0.1        # Quick-2023 early-stop lr-ratio threshold
BISECT_LOWER = 0.0
BISECT_UPPER = 0.1

# Feasibility gate (mirrors benchmarks/dei_layout.py, reimplemented here)
BOUNDARY_TOL = 1e-3
SPACING_SLACK = 0.99

# Host-side gradient-evaluation counter (jax.debug.callback)
_exec_counter = {"calls": 0, "ndims": set()}


def _count_cb(x):
    _exec_counter["calls"] += 1
    _exec_counter["ndims"].add(int(np.ndim(x)))


def compute_activation_step(mid, n_const, decay_len, threshold=ES_THRESHOLD):
    """First absolute step at which the compounded lr ratio <= threshold.

    lr after t decay steps: lr0 * prod_{k=1..t} 1/(1 + mid*k). Returns
    C + t for the first such t, or None if not reached within decay_len.
    """
    ratio = 1.0
    for t in range(1, decay_len + 1):
        ratio *= 1.0 / (1.0 + mid * t)
        if ratio <= threshold:
            return n_const + t
    return None


def load_problem(path):
    info = json.load(open(path))
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

    return objective, boundary, n_target, min_spacing


def grid_candidates(boundary, min_spacing):
    """Grid points inside the polygon (same logic as run_single_baseline.py)."""
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    nx = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing / 2, x_max - min_spacing / 2, nx),
        jnp.linspace(y_min + min_spacing / 2, y_max - min_spacing / 2, ny))
    cand_x, cand_y = gx.flatten(), gy.flatten()
    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    return cand_x[inside], cand_y[inside], (x_min, x_max, y_min, y_max)


def make_init_batch(seed, n_starts, boundary, n_target, min_spacing):
    """K grid-subsample initial layouts; PRNGKey(seed*1000 + lane) per lane."""
    inside_x, inside_y, (x_min, x_max, y_min, y_max) = grid_candidates(
        boundary, min_spacing)
    xs, ys = [], []
    for lane in range(n_starts):
        key = jax.random.PRNGKey(seed * 1000 + lane)
        if len(inside_x) >= n_target:
            idx = jax.random.choice(key, len(inside_x), (n_target,), replace=False)
            xs.append(inside_x[idx])
            ys.append(inside_y[idx])
        else:
            init_x = jax.random.uniform(key, (n_target,),
                                        minval=float(x_min), maxval=float(x_max))
            key, _ = jax.random.split(key)
            init_y = jax.random.uniform(key, (n_target,),
                                        minval=float(y_min), maxval=float(y_max))
            xs.append(init_x)
            ys.append(init_y)
    return jnp.stack(xs), jnp.stack(ys)


def interpret_counter(n_starts, iters):
    """Turn raw callback counts into evals_executed, or None if unreliable.

    Per lane, the instrumented objective is invoked once for state init,
    once per executed while_loop body iteration, and once for the final
    all_objs evaluation — i.e. batched_iters = calls_per_batch - 2.
    Under vmap the callback may fire once per batched call (arg ndim == 2)
    or once per lane (arg ndim == 1); both are handled.
    """
    raw = _exec_counter["calls"]
    ndims = _exec_counter["ndims"]
    if raw == 0:
        return None, None, "callback_unreliable: no callbacks fired"
    if ndims == {2}:
        calls_per_batch = raw
    elif ndims == {1}:
        if raw % n_starts != 0:
            return None, None, (
                f"callback_unreliable: {raw} per-lane calls not divisible "
                f"by K={n_starts}")
        calls_per_batch = raw // n_starts
    else:
        return None, None, f"callback_unreliable: mixed arg ndims {sorted(ndims)}"
    batched_iters = calls_per_batch - 2
    if not (1 <= batched_iters <= iters):
        return None, None, (
            f"callback_unreliable: inferred {batched_iters} batched iters "
            f"outside [1, {iters}]")
    return batched_iters, batched_iters * n_starts, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("problem", help="Path to problem JSON")
    p.add_argument("--starts", type=int, required=True,
                   help="K vmapped multistarts")
    p.add_argument("--budget", type=int, default=8000,
                   help="Total gradient-evaluation budget (default 8000)")
    p.add_argument("--iters", type=int, default=None,
                   help="Iterations per start T (default budget // starts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-es", action="store_true",
                   help="Disable Quick-2023 early stopping")
    p.add_argument("--out", default=None, help="Also write JSON result here")
    p.add_argument("--save-layout", default=None,
                   help="Write best layout {x, y, ...} JSON here")
    args = p.parse_args()

    K = args.starts
    T = args.iters if args.iters is not None else args.budget // K
    if T < 2:
        p.error(f"iters per start T={T} too small (budget {args.budget}, K={K})")

    objective, boundary, n_target, min_spacing = load_problem(args.problem)

    # Instrumented objective for the solver only (host-side eval counter).
    def counted_objective(x, y):
        jax.debug.callback(_count_cb, x)
        return objective(x, y)

    # --- Schedule scaled to fit T -------------------------------------------
    C = round(CONST_FRAC * T)
    M = round(DECAY_FRAC * T)
    mid = _compute_mid_bisection(
        learning_rate=LR0, gamma_min=GAMMA_MIN, max_iter=M,
        lower=BISECT_LOWER, upper=BISECT_UPPER)
    activation_step = compute_activation_step(mid, C, M)

    settings = SGDSettings(
        learning_rate=LR0,
        gamma_min_factor=GAMMA_MIN,
        max_iter=T - C,                          # decay-phase iteration cap
        additional_constant_lr_iterations=C,     # total_iter = T exactly
        mid=mid,                                 # pins the M-step decay shape
        beta1=0.1,
        beta2=0.2,
        early_stopping=not args.no_es,
        early_stop_threshold=ES_THRESHOLD,
    )

    init_x, init_y = make_init_batch(args.seed, K, boundary, n_target, min_spacing)

    t0 = time.time()
    all_x, all_y, all_objs = topfarm_sgd_solve_multistart(
        counted_objective, init_x, init_y, boundary, min_spacing, settings)
    all_x.block_until_ready()
    all_y.block_until_ready()
    all_objs.block_until_ready()
    try:
        jax.effects_barrier()  # flush pending debug callbacks
    except AttributeError:
        pass
    elapsed = time.time() - t0

    # --- Per-lane feasibility + AEP -----------------------------------------
    per_lane = []
    eye = jnp.eye(n_target) * 1e10
    for lane in range(K):
        x, y = all_x[lane], all_y[lane]
        aep = float(-all_objs[lane])
        bnd = float(boundary_penalty(x, y, boundary))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        min_dist = float(jnp.min(jnp.sqrt(dx**2 + dy**2 + eye)))
        feasible = (bnd < BOUNDARY_TOL) and (min_dist >= SPACING_SLACK * min_spacing)
        per_lane.append({
            "aep_gwh": round(aep, 3),
            "feasible": feasible,
            "boundary_penalty": bnd,
            "min_dist_m": round(min_dist, 2),
        })

    feasible_lanes = [i for i, r in enumerate(per_lane) if r["feasible"]]
    if feasible_lanes:
        best_lane = max(feasible_lanes, key=lambda i: per_lane[i]["aep_gwh"])
        best_feasible = True
    else:
        best_lane = max(range(K), key=lambda i: per_lane[i]["aep_gwh"])
        best_feasible = False

    batched_iters, evals_executed, callback_note = interpret_counter(K, T)

    result = {
        "aep_gwh": per_lane[best_lane]["aep_gwh"],
        "feasible": best_feasible,
        "best_lane": best_lane,
        "n_feasible": len(feasible_lanes),
        "n_starts": K,
        "iters": T,
        "budget": args.budget,
        "evals_billed": K * T,
        "C": C,
        "M": M,
        "mid": mid,
        "activation_step": activation_step,
        "es_enabled": not args.no_es,
        "seed": args.seed,
        "time_s": round(elapsed, 1),
        "per_lane": per_lane,
    }
    if evals_executed is not None:
        result["evals_executed"] = evals_executed
        result["batched_iters_executed"] = batched_iters
    else:
        result["callback_note"] = callback_note

    out_json = json.dumps(result, indent=2)
    print(out_json)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_json + "\n")
    if args.save_layout:
        with open(args.save_layout, "w") as f:
            json.dump({
                "x": [float(v) for v in all_x[best_lane]],
                "y": [float(v) for v in all_y[best_lane]],
                "lane": best_lane,
                "aep_gwh": per_lane[best_lane]["aep_gwh"],
                "feasible": best_feasible,
                "problem": os.path.abspath(args.problem),
                "seed": args.seed,
            }, f, indent=2)


if __name__ == "__main__":
    main()
