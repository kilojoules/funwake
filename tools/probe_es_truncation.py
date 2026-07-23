#!/usr/bin/env python
"""Calibration probe for Quick 2023 Algorithm 1 early stopping (ES) in the
vendored pixwake SGD solver.

Measures:
  (a) the iteration at which ES activates (analytic prediction from the
      compounding lr decay law, cross-checked against measured eval counts),
  (b) the spread of the constraint-cleanup tail after activation,
  (c) feasibility of the truncated result,
  (d) AEP given up by ES vs a full run (same seed/init).

Iteration counts are measured by wrapping the objective passed to
topfarm_sgd_solve with a jax.debug.callback that bumps a host-side counter
(fires on the primal pass of jax.grad, once per while_loop iteration, plus
one init-gradient call before the loop -> evals ~= iterations + 1).

Usage:
    pixi run python tools/probe_es_truncation.py results/problem_dei_n50.json \
        --seeds 0-9 --full-seeds 0,1 \
        --out results/equiv_cost_sgd/es_calibration.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "dependencies", "pixwake", "src"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import (SGDSettings, topfarm_sgd_solve,
                               boundary_penalty, _compute_mid_bisection)

LR0 = 50.0
MAX_ITER = 4000          # decaying-lr steps
N_CONST = 2000           # constant-lr steps
TOTAL_ITER = MAX_ITER + N_CONST
ES_THRESHOLD = 0.1
GAMMA_MIN = 0.01         # absolute target lr after MAX_ITER decay steps


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class EvalCounter:
    """Host-side counter bumped from inside traced code via debug callback."""

    def __init__(self):
        self.n = 0

    def bump(self, _):
        self.n += 1


def effects_barrier():
    """Flush any outstanding callback effects (best effort across jax versions)."""
    try:
        jax.effects_barrier()
    except AttributeError:
        pass


def make_counted_objective(objective, counter):
    """Wrap objective so every (primal) evaluation bumps the counter.

    jax.debug.callback works under jax.grad and inside lax.while_loop; this is
    the mechanism used (io_callback is not differentiable and would fail under
    jax.grad). The sanity check below verifies the count empirically.
    """

    def counted(x, y):
        jax.debug.callback(counter.bump, jnp.sum(x))
        return objective(x, y)

    return counted


def load_problem(path):
    """Build objective + geometry from a problem JSON (mirrors
    tools/run_single_baseline.py)."""
    info = json.load(open(path))
    D = info["rotor_diameter"]
    t = info["turbine"]
    turb = Turbine(
        rotor_diameter=D, hub_height=info.get("hub_height", 150.0),
        power_curve=Curve(ws=jnp.array(t["power_curve_ws"], dtype=float),
                          values=jnp.array(t["power_curve_kw"], dtype=float)),
        ct_curve=Curve(ws=jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]),
                                    dtype=float),
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


def grid_subsample_init(seed, boundary, n_target, min_spacing):
    """Grid init with random subsampling (mirrors run_single_baseline.py)."""
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
    inside_x, inside_y = cand_x[inside], cand_y[inside]

    key = jax.random.PRNGKey(seed)
    if len(inside_x) >= n_target:
        idx = jax.random.choice(key, len(inside_x), (n_target,), replace=False)
        return inside_x[idx], inside_y[idx]
    init_x = jax.random.uniform(key, (n_target,), minval=float(x_min),
                                maxval=float(x_max))
    key, _ = jax.random.split(key)
    init_y = jax.random.uniform(key, (n_target,), minval=float(y_min),
                                maxval=float(y_max))
    return init_x, init_y


def analytic_activation():
    """First decay step t where the compounded lr ratio drops to <= threshold.

    Decay law (sgd.py _sgd_step): after decay step t,
        lr_t = lr0 * prod_{s=1..t} 1/(1 + mid*s)
    with mid from bisection so lr_{MAX_ITER} == GAMMA_MIN (absolute).
    ES checks state.learning_rate/lr0 at the START of each body iteration, so
    the first ES-active iteration is total step N_CONST + t + 1.
    """
    mid = _compute_mid_bisection(learning_rate=LR0, gamma_min=GAMMA_MIN,
                                 max_iter=MAX_ITER, lower=0.0, upper=0.1)
    ratio = 1.0
    t_act = None
    for t in range(1, MAX_ITER + 1):
        ratio *= 1.0 / (1.0 + mid * t)
        if ratio <= ES_THRESHOLD:
            t_act = t
            break
    return mid, t_act


def measure_layout(objective, x, y, boundary, min_spacing, n_target):
    aep = float(-objective(x, y))
    bnd_pen = float(boundary_penalty(x, y, boundary))
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(n_target) * 1e10)
    min_dist = float(jnp.min(dist))
    feasible = (bnd_pen < 1e-3) and (min_dist >= min_spacing * 0.99)
    return aep, bnd_pen, min_dist, feasible


def run_solve(objective, init_x, init_y, boundary, min_spacing, n_target,
              early_stopping):
    counter = EvalCounter()
    counted = make_counted_objective(objective, counter)
    settings = SGDSettings(learning_rate=LR0, max_iter=MAX_ITER,
                           additional_constant_lr_iterations=N_CONST,
                           beta1=0.1, beta2=0.2,
                           early_stopping=early_stopping,
                           early_stop_threshold=ES_THRESHOLD)
    t0 = time.time()
    opt_x, opt_y = topfarm_sgd_solve(counted, init_x, init_y, boundary,
                                     min_spacing, settings)
    opt_x.block_until_ready()
    opt_y.block_until_ready()
    effects_barrier()
    elapsed = time.time() - t0
    evals = counter.n
    aep, bnd_pen, min_dist, feasible = measure_layout(
        objective, opt_x, opt_y, boundary, min_spacing, n_target)
    return {
        "evals": evals,
        "aep_gwh": round(aep, 3),
        "boundary_penalty": bnd_pen,
        "min_spacing_m": round(min_dist, 2),
        "feasible": feasible,
        "wall_time_s": round(elapsed, 1),
        "hit_cap": evals >= TOTAL_ITER + 1,
    }


def sanity_check_counter(objective, boundary, n_target, min_spacing):
    """Tiny run (50 decay steps, no constant phase, no ES): expect ~51 evals."""
    init_x, init_y = grid_subsample_init(0, boundary, n_target, min_spacing)
    counter = EvalCounter()
    counted = make_counted_objective(objective, counter)
    settings = SGDSettings(learning_rate=LR0, max_iter=50,
                           additional_constant_lr_iterations=0,
                           beta1=0.1, beta2=0.2)
    x, y = topfarm_sgd_solve(counted, init_x, init_y, boundary, min_spacing,
                             settings)
    x.block_until_ready()
    y.block_until_ready()
    effects_barrier()
    return counter.n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="results/problem_dei_n50.json")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--full-seeds", default="0,1")
    p.add_argument("--out", default="results/equiv_cost_sgd/es_calibration.json")
    p.add_argument("--sanity-only", action="store_true",
                   help="Run only the eval-counter sanity check and exit.")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    full_seeds = parse_seeds(args.full_seeds)

    objective, boundary, n_target, min_spacing = load_problem(args.problem)

    # --- Step 1: eval-counter sanity check ------------------------------------
    log("[sanity] running tiny config (max_iter=50, no const phase, no ES)...")
    t0 = time.time()
    sanity_evals = sanity_check_counter(objective, boundary, n_target,
                                        min_spacing)
    log(f"[sanity] counter={sanity_evals} (expected ~51), "
        f"{time.time()-t0:.1f}s")
    sanity = {
        "expected_evals": 51,
        "observed_evals": sanity_evals,
        "ok": abs(sanity_evals - 51) <= 1,
        "mechanism": "jax.debug.callback",
    }
    if args.sanity_only:
        print(json.dumps({"sanity": sanity}, indent=2))
        return
    if not sanity["ok"]:
        log("[sanity] WARNING: counter off by more than 1; results may be "
            "unreliable. Proceeding anyway.")

    # --- Step 2: analytic activation step -------------------------------------
    mid, t_act = analytic_activation()
    activation = {
        "mid": mid,
        "t_activation_decay_step": t_act,
        "predicted_activation_total_step": N_CONST + t_act,
        "first_es_active_iteration": N_CONST + t_act + 1,
        "fraction_of_decay_phase": round(t_act / MAX_ITER, 4),
    }
    log(f"[analytic] mid={mid:.6g}, lr ratio <= {ES_THRESHOLD} after decay "
        f"step {t_act} (total step {N_CONST + t_act}, "
        f"{100*t_act/MAX_ITER:.1f}% of decay phase)")

    # --- Step 3: ES runs over seeds -------------------------------------------
    es_runs = []
    for seed in seeds:
        log(f"[es] seed {seed}...")
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        rec = run_solve(objective, init_x, init_y, boundary, min_spacing,
                        n_target, early_stopping=True)
        rec["seed"] = seed
        rec["tail"] = rec["evals"] - (N_CONST + t_act + 1)
        es_runs.append(rec)
        log(f"[es] seed {seed}: evals={rec['evals']} tail={rec['tail']} "
            f"aep={rec['aep_gwh']} feasible={rec['feasible']} "
            f"hit_cap={rec['hit_cap']} ({rec['wall_time_s']}s)")

    # --- Step 4: full runs for comparison seeds -------------------------------
    full_runs = []
    for seed in full_seeds:
        log(f"[full] seed {seed}...")
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        rec = run_solve(objective, init_x, init_y, boundary, min_spacing,
                        n_target, early_stopping=False)
        rec["seed"] = seed
        es_rec = next((r for r in es_runs if r["seed"] == seed), None)
        if es_rec is not None:
            rec["aep_delta_es_minus_full"] = round(
                es_rec["aep_gwh"] - rec["aep_gwh"], 3)
            rec["evals_es"] = es_rec["evals"]
        full_runs.append(rec)
        log(f"[full] seed {seed}: evals={rec['evals']} aep={rec['aep_gwh']} "
            f"delta_es={rec.get('aep_delta_es_minus_full')} "
            f"({rec['wall_time_s']}s)")

    # --- Step 5: summary + write ----------------------------------------------
    tails = [r["tail"] for r in es_runs]
    deltas = [r["aep_delta_es_minus_full"] for r in full_runs
              if "aep_delta_es_minus_full" in r]
    n_feas = sum(r["feasible"] for r in es_runs)
    n_cap = sum(r["hit_cap"] for r in es_runs)
    max_tail = max(tails)
    # Recommended fixed-cap cleanup allowance: max observed tail with 25%
    # safety margin, as a fraction of the total iteration budget.
    rec_frac = round(1.25 * max_tail / TOTAL_ITER, 4) if max_tail > 0 else 0.0
    summary = {
        "n_seeds": len(es_runs),
        "activation_analytic_total_step": N_CONST + t_act,
        "evals_min": min(r["evals"] for r in es_runs),
        "evals_median": float(np.median([r["evals"] for r in es_runs])),
        "evals_max": max(r["evals"] for r in es_runs),
        "tail_min": min(tails),
        "tail_median": float(np.median(tails)),
        "tail_max": max_tail,
        "feasibility_rate": n_feas / len(es_runs),
        "n_hit_cap": n_cap,
        "mean_aep_delta_es_minus_full_gwh": (
            round(float(np.mean(deltas)), 3) if deltas else None),
        "recommended_cleanup_allowance_frac_of_T": rec_frac,
    }

    out = {
        "problem": args.problem,
        "settings": {
            "learning_rate": LR0, "max_iter": MAX_ITER,
            "additional_constant_lr_iterations": N_CONST,
            "beta1": 0.1, "beta2": 0.2,
            "early_stop_threshold": ES_THRESHOLD,
            "gamma_min_factor": GAMMA_MIN,
            "total_iterations": TOTAL_ITER,
        },
        "counter_sanity": sanity,
        "activation": activation,
        "es_runs": es_runs,
        "full_runs": full_runs,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"[done] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
