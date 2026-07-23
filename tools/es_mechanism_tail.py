#!/usr/bin/env python
"""ES-mechanism experiment 2: tail trajectory reconstruction.

Replays the full-run SGD trajectory with a lax.scan built from the solver's
own `_sgd_step` and gradient definitions (imported from pixwake.optim.sgd,
solver untouched), capturing turbine positions at every step. Verifies
bit-identity against capped `topfarm_sgd_solve` runs (mid pinned) at the ES
activation point (step 4079) and the full run (step 6000).

From the trajectory it computes, per step of the tail (4000..6000):
  raw AEP, boundary penalty, min spacing, min signed boundary distance,
  n turbines within 50 m of the boundary, lr(t), alpha(t)
plus the requested sparse cap grid and per-turbine tail kinematics
(net displacement vs path length -> oscillation index).

Counterfactual tail variants from the exact step-4079 state:
  baseline  : combined gradient (identical to full run tail)
  es_like   : AEP gradient zeroed (what ES would do if it kept stepping)
  aep_only  : constraint gradient zeroed
  frozen    : combined gradient but lr/alpha frozen at step-4079 values
These separate "AEP-step dynamics lose AEP" from "boundary forces lose AEP"
from "the annealing itself loses AEP".

Usage:
    pixi run python tools/es_mechanism_tail.py --seeds 0,1 \
        --out results/equiv_cost_sgd/es_mechanism/tail_curves.json
"""
import argparse
import json
import os
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from probe_es_truncation import (LR0, MAX_ITER, N_CONST, ES_THRESHOLD,
                                 GAMMA_MIN, load_problem, grid_subsample_init,
                                 parse_seeds, log)

import numpy as np
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import (SGDSettings, topfarm_sgd_solve, _sgd_step,
                               _init_sgd_state, boundary_penalty,
                               spacing_penalty, _compute_mid_bisection)

T_ACT = 4079          # last step BEFORE ES becomes active (activation at 4080)
T_TOTAL = 6000
CAP_GRID = [4079, 4080, 4090, 4110, 4150, 4250, 4400, 4600, 4900, 5300, 6000]
TAIL_START = 4000     # dense curves start a little before activation
BOUNDARY_NEAR_M = 50.0


def canon_state(state):
    """Canonicalize SGDState fields to arrays for scan carry stability."""
    return state._replace(
        iteration=jnp.asarray(state.iteration),
        learning_rate=jnp.asarray(state.learning_rate),
        alpha=jnp.asarray(state.alpha),
        alpha0=jnp.asarray(state.alpha0),
        lr0=jnp.asarray(state.lr0),
    )


def make_scan(objective, boundary, min_spacing, settings, mode="baseline"):
    """Build a jitted scan over n steps of the solver's update law.

    mode: baseline | es_like (zero AEP grad) | aep_only (zero constraint
    grad) | frozen (combined grad, but lr/alpha reset to entry values after
    each step, i.e. no further annealing).
    """
    def constraint_penalty(x, y):
        return (settings.boundary_weight
                * boundary_penalty(x, y, boundary, settings.ks_rho)
                + settings.spacing_weight
                * spacing_penalty(x, y, min_spacing, settings.ks_rho))

    grad_obj_fn = jax.grad(objective, argnums=(0, 1))
    grad_con_fn = jax.grad(constraint_penalty, argnums=(0, 1))

    def body(carry, _):
        x, y, state = carry
        gox, goy = grad_obj_fn(x, y)
        gcx, gcy = grad_con_fn(x, y)
        if mode == "es_like":
            gox = jnp.zeros_like(gox)
            goy = jnp.zeros_like(goy)
        elif mode == "aep_only":
            gcx = jnp.zeros_like(gcx)
            gcy = jnp.zeros_like(gcy)
        xn, yn, sn = _sgd_step(x, y, state, gox, goy, gcx, gcy, settings)
        if mode == "frozen":
            sn = sn._replace(learning_rate=state.learning_rate,
                             alpha=state.alpha)
        return (xn, yn, sn), (xn, yn)

    from functools import partial

    @partial(jax.jit, static_argnums=(3,))
    def run(x, y, state, n):
        (xf, yf, sf), (xs, ys) = jax.lax.scan(body, (x, y, state), None,
                                              length=n)
        return xf, yf, sf, xs, ys

    return run


def signed_boundary_dist(xs, ys, boundary):
    """Per-turbine signed distance to convex polygon (positive inside).

    xs, ys: (..., n_turbines). Returns same shape.
    """
    bv = np.asarray(boundary)
    n_verts = bv.shape[0]
    dists = []
    for i in range(n_verts):
        x1, y1 = bv[i]
        x2, y2 = bv[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = np.sqrt(ex**2 + ey**2) + 1e-10
        dists.append((xs - x1) * (-ey / el) + (ys - y1) * (ex / el))
    return np.min(np.stack(dists, axis=0), axis=0)


def min_spacing_per_step(xs, ys):
    """xs, ys: (T, n). Returns (T,) min pairwise distance per step."""
    dx = xs[:, :, None] - xs[:, None, :]
    dy = ys[:, :, None] - ys[:, None, :]
    d = np.sqrt(dx**2 + dy**2)
    n = xs.shape[1]
    d[:, np.arange(n), np.arange(n)] = np.inf
    return d.min(axis=(1, 2))


def batched_aep(objective, xs, ys, chunk=32):
    """Raw AEP (GWh) for a stack of layouts, vmapped in chunks."""
    f = jax.jit(jax.vmap(lambda x, y: -objective(x, y)))
    out = []
    for i in range(0, xs.shape[0], chunk):
        out.append(np.asarray(f(jnp.asarray(xs[i:i + chunk]),
                                jnp.asarray(ys[i:i + chunk]))))
    return np.concatenate(out)


def batched_boundary_penalty(xs, ys, boundary, chunk=256):
    f = jax.jit(jax.vmap(lambda x, y: boundary_penalty(x, y, boundary)))
    out = []
    for i in range(0, xs.shape[0], chunk):
        out.append(np.asarray(f(jnp.asarray(xs[i:i + chunk]),
                                jnp.asarray(ys[i:i + chunk]))))
    return np.concatenate(out)


def lr_alpha_schedule(mid, alpha0):
    """Analytic lr(t), alpha(t) AFTER step t (t = 1..6000)."""
    lr = np.empty(T_TOTAL + 1)
    lr[0] = LR0
    for it in range(1, T_TOTAL + 1):
        decay_it = max(it - N_CONST, 0)
        lr[it] = lr[it - 1] / (1.0 + mid * decay_it)
    alpha = np.where(np.arange(T_TOTAL + 1) > N_CONST,
                     alpha0 * LR0 / lr, alpha0)
    return lr, alpha


def analyze_seed(seed, objective, boundary, min_spacing, n_target, mid,
                 verify):
    settings = SGDSettings(learning_rate=LR0, max_iter=MAX_ITER,
                           additional_constant_lr_iterations=N_CONST,
                           beta1=0.1, beta2=0.2, mid=mid,
                           early_stopping=False)
    init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                         min_spacing)

    # --- Phase 1: steps 1..T_ACT ------------------------------------------
    grad_obj_fn = jax.grad(objective, argnums=(0, 1))
    g0x, g0y = grad_obj_fn(init_x, init_y)
    state0 = canon_state(_init_sgd_state(init_x, init_y, g0x, g0y, settings))
    alpha0 = float(state0.alpha0)

    run = make_scan(objective, boundary, min_spacing, settings, "baseline")
    t0 = time.time()
    x_act, y_act, s_act, xs1, ys1 = run(init_x, init_y, state0, T_ACT)
    x_act.block_until_ready()
    log(f"[seed {seed}] phase1 ({T_ACT} steps) {time.time()-t0:.1f}s "
        f"lr@act={float(s_act.learning_rate):.4f} "
        f"alpha@act={float(s_act.alpha):.4f}")

    # --- Phase 2: tail variants from the exact step-4079 state ------------
    n_tail = T_TOTAL - T_ACT
    variants = {}
    for mode in ["baseline", "es_like", "aep_only", "frozen"]:
        runv = make_scan(objective, boundary, min_spacing, settings, mode)
        t0 = time.time()
        xf, yf, sf, xs2, ys2 = runv(x_act, y_act, s_act, n_tail)
        xf.block_until_ready()
        variants[mode] = (np.asarray(xs2), np.asarray(ys2))
        log(f"[seed {seed}] tail variant {mode} ({n_tail} steps) "
            f"{time.time()-t0:.1f}s")

    # Full trajectory including step 0 (positions AFTER step t at index t)
    xs = np.concatenate([np.asarray(init_x)[None], np.asarray(xs1),
                         variants["baseline"][0]])
    ys = np.concatenate([np.asarray(init_y)[None], np.asarray(ys1),
                         variants["baseline"][1]])
    assert xs.shape == (T_TOTAL + 1, n_target)

    # --- Verification against the actual solver (capped replays) ----------
    verification = {}
    if verify:
        for cap in [T_ACT, T_TOTAL]:
            s = SGDSettings(learning_rate=LR0, max_iter=cap - N_CONST,
                            additional_constant_lr_iterations=N_CONST,
                            beta1=0.1, beta2=0.2, mid=mid,
                            early_stopping=False)
            t0 = time.time()
            vx, vy = topfarm_sgd_solve(objective, init_x, init_y, boundary,
                                       min_spacing, s)
            vx.block_until_ready()
            dmax = float(max(np.max(np.abs(np.asarray(vx) - xs[cap])),
                             np.max(np.abs(np.asarray(vy) - ys[cap]))))
            verification[f"cap_{cap}_max_abs_diff_m"] = dmax
            log(f"[seed {seed}] verify cap={cap}: max|diff|={dmax:.3e} m "
                f"({time.time()-t0:.1f}s)")

    # --- Dense curves ------------------------------------------------------
    lr, alpha = lr_alpha_schedule(mid, alpha0)

    # Tail: every step. Pre-tail context: every 20 steps.
    tail_steps = np.arange(TAIL_START, T_TOTAL + 1)
    pre_steps = np.arange(0, TAIL_START, 20)

    t0 = time.time()
    aep_tail = batched_aep(objective, xs[tail_steps], ys[tail_steps])
    aep_pre = batched_aep(objective, xs[pre_steps], ys[pre_steps])
    log(f"[seed {seed}] {len(tail_steps)+len(pre_steps)} AEP evals "
        f"{time.time()-t0:.1f}s")

    pen_tail = batched_boundary_penalty(xs[tail_steps], ys[tail_steps],
                                        boundary)
    minsp_tail = min_spacing_per_step(xs[tail_steps], ys[tail_steps])
    bdist_tail = signed_boundary_dist(xs[tail_steps], ys[tail_steps],
                                      boundary)  # (T, n)
    n_near_tail = (bdist_tail < BOUNDARY_NEAR_M).sum(axis=1)
    min_bdist_tail = bdist_tail.min(axis=1)

    # Variant AEP curves (every 10 steps of the tail)
    var_stride = 10
    var_steps = np.arange(T_ACT, T_TOTAL + 1, var_stride)
    variant_curves = {}
    for mode, (vxs, vys) in variants.items():
        vx_full = np.concatenate([xs[T_ACT][None], vxs])
        vy_full = np.concatenate([ys[T_ACT][None], vys])
        idx = var_steps - T_ACT
        variant_curves[mode] = {
            "steps": var_steps.tolist(),
            "aep_gwh": np.round(batched_aep(objective, vx_full[idx],
                                            vy_full[idx]), 4).tolist(),
            "final_boundary_penalty": float(
                boundary_penalty(jnp.asarray(vx_full[-1]),
                                 jnp.asarray(vy_full[-1]), boundary)),
        }

    # --- Sparse cap grid ---------------------------------------------------
    caps = []
    for cap in CAP_GRID:
        i = cap - TAIL_START
        caps.append({
            "cap": cap,
            "aep_gwh": round(float(aep_tail[i]), 4),
            "boundary_penalty": float(pen_tail[i]),
            "min_spacing_m": round(float(minsp_tail[i]), 2),
            "lr": float(lr[cap]),
            "alpha_over_alpha0": float(alpha[cap] / alpha0),
            "n_within_50m_of_boundary": int(n_near_tail[i]),
            "min_signed_boundary_dist_m": round(float(min_bdist_tail[i]), 3),
        })

    # --- Per-turbine tail kinematics ---------------------------------------
    xs_tail = xs[T_ACT:]
    ys_tail = ys[T_ACT:]
    net_disp = np.sqrt((xs_tail[-1] - xs_tail[0])**2
                       + (ys_tail[-1] - ys_tail[0])**2)
    seg = np.sqrt(np.diff(xs_tail, axis=0)**2 + np.diff(ys_tail, axis=0)**2)
    path_len = seg.sum(axis=0)
    # Oscillation index: path length / max(net displacement, 1 m)
    osc = path_len / np.maximum(net_disp, 1.0)
    bdist_act = signed_boundary_dist(xs[T_ACT], ys[T_ACT], boundary)
    bdist_end = signed_boundary_dist(xs[T_TOTAL], ys[T_TOTAL], boundary)

    # AEP loss localization within the tail
    i_act = T_ACT - TAIL_START
    aep_act = float(aep_tail[i_act])
    aep_end = float(aep_tail[-1])
    aep_min = float(aep_tail[i_act:].min())
    t_min = int(tail_steps[i_act + int(np.argmin(aep_tail[i_act:]))])
    # first step in tail where AEP has irrecoverably dropped below final
    below = np.where(aep_tail[i_act:] < aep_end)[0]

    return {
        "seed": seed,
        "alpha0": alpha0,
        "lr_at_activation": float(lr[T_ACT]),
        "verification": verification,
        "caps": caps,
        "dense_tail": {
            "steps": tail_steps.tolist(),
            "aep_gwh": np.round(aep_tail, 4).tolist(),
            "boundary_penalty": [float(v) for v in pen_tail],
            "min_spacing_m": np.round(minsp_tail, 2).tolist(),
            "n_within_50m_of_boundary": n_near_tail.tolist(),
            "min_signed_boundary_dist_m": np.round(min_bdist_tail,
                                                   3).tolist(),
            "lr": np.round(lr[tail_steps], 5).tolist(),
        },
        "dense_pre": {
            "steps": pre_steps.tolist(),
            "aep_gwh": np.round(aep_pre, 4).tolist(),
        },
        "variant_tails": variant_curves,
        "tail_summary": {
            "aep_at_activation": round(aep_act, 4),
            "aep_at_end": round(aep_end, 4),
            "aep_tail_delta": round(aep_end - aep_act, 4),
            "aep_tail_min": round(aep_min, 4),
            "step_of_tail_min": t_min,
            "first_tail_step_below_final": (
                int(tail_steps[i_act + below[0]]) if len(below) else None),
        },
        "turbine_kinematics": {
            "net_disp_m": np.round(net_disp, 3).tolist(),
            "path_len_m": np.round(path_len, 1).tolist(),
            "oscillation_index": np.round(osc, 1).tolist(),
            "boundary_dist_at_activation_m": np.round(bdist_act, 2).tolist(),
            "boundary_dist_at_end_m": np.round(bdist_end, 2).tolist(),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "tail_curves.json"))
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    objective, boundary, n_target, min_spacing = load_problem(args.problem)
    mid = _compute_mid_bisection(learning_rate=LR0, gamma_min=GAMMA_MIN,
                                 max_iter=MAX_ITER, lower=0.0, upper=0.1)
    log(f"[setup] mid={mid!r} seeds={seeds}")

    results = []
    for seed in seeds:
        results.append(analyze_seed(seed, objective, boundary, min_spacing,
                                    n_target, mid, verify=not args.no_verify))

    out = {
        "problem": args.problem,
        "mid": mid,
        "t_activation": T_ACT,
        "t_total": T_TOTAL,
        "cap_grid": CAP_GRID,
        "seeds": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Merge with existing file if present (so seeds can be added later).
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            have = {r["seed"] for r in results}
            for r in prev.get("seeds", []):
                if r["seed"] not in have:
                    out["seeds"].append(r)
            out["seeds"].sort(key=lambda r: r["seed"])
        except (json.JSONDecodeError, KeyError):
            pass
    with open(args.out, "w") as f:
        json.dump(out, f)
    log(f"[done] wrote {args.out}")
    for r in out["seeds"]:
        print(json.dumps({"seed": r["seed"], **r["tail_summary"],
                          "verification": r["verification"]}, indent=2))


if __name__ == "__main__":
    main()
