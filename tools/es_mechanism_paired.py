#!/usr/bin/env python
"""ES-mechanism experiment 1: paired ES vs full runs on all 10 seeds.

For each seed, runs the SGD solver twice from an identical init:
  - full: early_stopping=False (6000 steps: 2000 const-lr + 4000 decaying)
  - es:   early_stopping=True, threshold 0.1 (activates at total step 4080)

Records raw AEP, boundary penalty, min spacing, and final layouts for both,
plus delta = AEP_ES - AEP_full per seed and summary stats.

`mid` is pinned to the bisection value so runs are directly comparable with
the capped-replay tooling (pinning changes nothing: it equals the value the
solver would compute internally).

Usage:
    pixi run python tools/es_mechanism_paired.py \
        --out results/equiv_cost_sgd/es_mechanism/paired_10seeds.json
"""
import argparse
import json
import os
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

# probe_es_truncation sets up sys.path for pixwake and enables jax x64.
from probe_es_truncation import (LR0, MAX_ITER, N_CONST, ES_THRESHOLD,
                                 GAMMA_MIN, load_problem, grid_subsample_init,
                                 measure_layout, parse_seeds, log)

import numpy as np
import jax.numpy as jnp
from pixwake.optim.sgd import (SGDSettings, topfarm_sgd_solve,
                               _compute_mid_bisection)


def solve(objective, init_x, init_y, boundary, min_spacing, n_target, mid,
          early_stopping):
    settings = SGDSettings(learning_rate=LR0, max_iter=MAX_ITER,
                           additional_constant_lr_iterations=N_CONST,
                           beta1=0.1, beta2=0.2, mid=mid,
                           early_stopping=early_stopping,
                           early_stop_threshold=ES_THRESHOLD)
    t0 = time.time()
    ox, oy = topfarm_sgd_solve(objective, init_x, init_y, boundary,
                               min_spacing, settings)
    ox.block_until_ready()
    oy.block_until_ready()
    elapsed = time.time() - t0
    aep, bnd_pen, min_dist, feasible = measure_layout(
        objective, ox, oy, boundary, min_spacing, n_target)
    return {
        "aep_gwh": aep,
        "boundary_penalty": bnd_pen,
        "min_spacing_m": round(min_dist, 3),
        "feasible": feasible,
        "wall_time_s": round(elapsed, 1),
        "x": np.asarray(ox).tolist(),
        "y": np.asarray(oy).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "paired_10seeds.json"))
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    objective, boundary, n_target, min_spacing = load_problem(args.problem)
    mid = _compute_mid_bisection(learning_rate=LR0, gamma_min=GAMMA_MIN,
                                 max_iter=MAX_ITER, lower=0.0, upper=0.1)
    log(f"[setup] mid={mid!r}")

    records = []
    for seed in seeds:
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        full = solve(objective, init_x, init_y, boundary, min_spacing,
                     n_target, mid, early_stopping=False)
        es = solve(objective, init_x, init_y, boundary, min_spacing,
                   n_target, mid, early_stopping=True)
        delta = es["aep_gwh"] - full["aep_gwh"]
        rec = {"seed": seed, "full": full, "es": es,
               "delta_es_minus_full_gwh": delta}
        records.append(rec)
        log(f"[seed {seed}] full={full['aep_gwh']:.3f} es={es['aep_gwh']:.3f} "
            f"delta={delta:+.3f} pen_full={full['boundary_penalty']:.3e} "
            f"pen_es={es['boundary_penalty']:.3e} "
            f"({full['wall_time_s']}+{es['wall_time_s']}s)")

    deltas = np.array([r["delta_es_minus_full_gwh"] for r in records])
    summary = {
        "n_seeds": len(records),
        "delta_mean_gwh": float(np.mean(deltas)),
        "delta_std_gwh": float(np.std(deltas)),
        "delta_min_gwh": float(np.min(deltas)),
        "delta_max_gwh": float(np.max(deltas)),
        "delta_median_gwh": float(np.median(deltas)),
        "n_es_higher": int(np.sum(deltas > 0)),
        "n_full_higher": int(np.sum(deltas < 0)),
        "largest_abs_delta_seed": int(
            records[int(np.argmax(np.abs(deltas)))]["seed"]),
        "mean_boundary_penalty_full": float(np.mean(
            [r["full"]["boundary_penalty"] for r in records])),
        "mean_boundary_penalty_es": float(np.mean(
            [r["es"]["boundary_penalty"] for r in records])),
    }

    out = {
        "problem": args.problem,
        "settings": {"learning_rate": LR0, "max_iter": MAX_ITER,
                     "additional_constant_lr_iterations": N_CONST,
                     "beta1": 0.1, "beta2": 0.2, "mid": mid,
                     "early_stop_threshold": ES_THRESHOLD},
        "seeds": records,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    log(f"[done] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
