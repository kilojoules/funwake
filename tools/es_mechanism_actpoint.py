#!/usr/bin/env python
"""ES-mechanism supplement: activation-point decomposition for all seeds.

Replays the (bit-verified) trajectory to the ES activation point (step 4079)
for every seed and decomposes the paired delta:

    delta = AEP_ES - AEP_full
          = (AEP_ES - AEP_act)  [ES cleanup effect: 2-4 constraint-only
                                 momentum-flush steps from the activation
                                 state]
          + (AEP_act - AEP_full) [negative of the full run's annealed-tail
                                  drift over steps 4079..6000]

Also records boundary state at activation (penalty, min signed distance,
n turbines outside) to characterize the lr~5 boundary limit cycle.

Usage:
    pixi run python tools/es_mechanism_actpoint.py \
        --out results/equiv_cost_sgd/es_mechanism/activation_decomposition.json
"""
import argparse
import json
import os
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from probe_es_truncation import (LR0, MAX_ITER, N_CONST, GAMMA_MIN,
                                 load_problem, grid_subsample_init,
                                 parse_seeds, log)
from es_mechanism_tail import (T_ACT, canon_state, make_scan,
                               signed_boundary_dist)

import numpy as np
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import (SGDSettings, _init_sgd_state,
                               boundary_penalty, _compute_mid_bisection)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--paired",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "paired_10seeds.json"))
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "activation_decomposition.json"))
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    paired = {r["seed"]: r for r in json.load(open(args.paired))["seeds"]}
    objective, boundary, n_target, min_spacing = load_problem(args.problem)
    mid = _compute_mid_bisection(learning_rate=LR0, gamma_min=GAMMA_MIN,
                                 max_iter=MAX_ITER, lower=0.0, upper=0.1)
    settings = SGDSettings(learning_rate=LR0, max_iter=MAX_ITER,
                           additional_constant_lr_iterations=N_CONST,
                           beta1=0.1, beta2=0.2, mid=mid,
                           early_stopping=False)
    grad_obj_fn = jax.grad(objective, argnums=(0, 1))
    run = make_scan(objective, boundary, min_spacing, settings, "baseline")

    rows = []
    for seed in seeds:
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        g0x, g0y = grad_obj_fn(init_x, init_y)
        state0 = canon_state(_init_sgd_state(init_x, init_y, g0x, g0y,
                                             settings))
        t0 = time.time()
        x_act, y_act, s_act, _, _ = run(init_x, init_y, state0, T_ACT)
        x_act.block_until_ready()
        aep_act = float(-objective(x_act, y_act))
        pen_act = float(boundary_penalty(x_act, y_act, boundary))
        bd = signed_boundary_dist(np.asarray(x_act), np.asarray(y_act),
                                  boundary)
        pr = paired[seed]
        aep_es = pr["es"]["aep_gwh"]
        aep_full = pr["full"]["aep_gwh"]
        row = {
            "seed": seed,
            "aep_act": round(aep_act, 4),
            "aep_es": round(aep_es, 4),
            "aep_full": round(aep_full, 4),
            "es_cleanup_effect": round(aep_es - aep_act, 4),
            "full_tail_drift": round(aep_full - aep_act, 4),
            "delta_es_minus_full": round(aep_es - aep_full, 4),
            "pen_act_m2": round(pen_act, 3),
            "n_outside_at_act": int((bd < 0).sum()),
            "min_signed_bdist_act_m": round(float(bd.min()), 3),
            "n_within_50m": int((bd < 50.0).sum()),
        }
        rows.append(row)
        log(f"[seed {seed}] act={aep_act:.3f} cleanup={row['es_cleanup_effect']:+.3f} "
            f"tail_drift={row['full_tail_drift']:+.3f} "
            f"delta={row['delta_es_minus_full']:+.3f} pen_act={pen_act:.1f} "
            f"n_out={row['n_outside_at_act']} ({time.time()-t0:.1f}s)")

    cleanup = np.array([r["es_cleanup_effect"] for r in rows])
    drift = np.array([r["full_tail_drift"] for r in rows])
    summary = {
        "cleanup_mean": round(float(cleanup.mean()), 4),
        "cleanup_std": round(float(cleanup.std()), 4),
        "cleanup_n_positive": int((cleanup > 0).sum()),
        "tail_drift_mean": round(float(drift.mean()), 4),
        "tail_drift_std": round(float(drift.std()), 4),
        "tail_drift_n_positive": int((drift > 0).sum()),
        "pen_act_mean_m2": round(float(np.mean(
            [r["pen_act_m2"] for r in rows])), 2),
        "mean_n_outside_at_act": round(float(np.mean(
            [r["n_outside_at_act"] for r in rows])), 1),
    }
    out = {"problem": args.problem, "t_act": T_ACT, "rows": rows,
           "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    log(f"[done] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
