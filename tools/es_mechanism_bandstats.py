#!/usr/bin/env python
"""ES-mechanism supplement: AEP noise-band statistics over the tail.

From tail_curves.json (dense per-step AEP over steps 4000..6000):
  - fast component: std of step-to-step AEP differences, per lr window;
  - slow component: window std of AEP itself;
  - percentile of the full-run endpoint and the ES endpoint within the
    tail band;
  - stationarity check of the pre-activation band (constant-lr phase).

Plus a direct freeze check: per-turbine displacement between the paired ES
final layout and the bit-verified activation-point (step 4079) layout.

Usage:
    pixi run python tools/es_mechanism_bandstats.py --seeds 0,1,5 \
        --out results/equiv_cost_sgd/es_mechanism/band_stats.json
"""
import argparse
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from probe_es_truncation import (LR0, MAX_ITER, N_CONST, GAMMA_MIN,
                                 load_problem, grid_subsample_init,
                                 parse_seeds, log)
from es_mechanism_tail import T_ACT, canon_state, make_scan

import numpy as np
import jax
from pixwake.optim.sgd import (SGDSettings, _init_sgd_state,
                               _compute_mid_bisection)

WINDOWS = [(4000, 4079), (4079, 4300), (4300, 4700), (4700, 5300),
           (5300, 5800), (5800, 6001)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--seeds", default="0,1,5")
    p.add_argument("--tail",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "tail_curves.json"))
    p.add_argument("--paired",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "paired_10seeds.json"))
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "band_stats.json"))
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    tail = {r["seed"]: r for r in json.load(open(args.tail))["seeds"]}
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

    out_rows = []
    for seed in seeds:
        r = tail[seed]
        steps = np.array(r["dense_tail"]["steps"])
        aep = np.array(r["dense_tail"]["aep_gwh"])
        lr = np.array(r["dense_tail"]["lr"])

        wins = []
        for lo, hi in WINDOWS:
            m = (steps >= lo) & (steps < hi)
            a = aep[m]
            wins.append({
                "window": [lo, hi],
                "lr_mid": round(float(lr[m].mean()), 3),
                "band_mean": round(float(a.mean()), 3),
                "band_std_slow": round(float(a.std()), 3),
                "step_diff_std_fast": round(float(np.diff(a).std()), 3),
                "band_p2p": round(float(a.max() - a.min()), 3),
            })

        # Endpoint percentiles within the tail band (steps 4079..6000)
        band = aep[steps >= T_ACT]
        aep_full = paired[seed]["full"]["aep_gwh"]
        aep_es = paired[seed]["es"]["aep_gwh"]
        pct_full = float((band < aep_full).mean() * 100)
        pct_es = float((band < aep_es).mean() * 100)

        # Freeze check: replay to activation, compare with paired ES layout
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        g0x, g0y = grad_obj_fn(init_x, init_y)
        state0 = canon_state(_init_sgd_state(init_x, init_y, g0x, g0y,
                                             settings))
        x_act, y_act, _, _, _ = run(init_x, init_y, state0, T_ACT)
        ex = np.array(paired[seed]["es"]["x"])
        ey = np.array(paired[seed]["es"]["y"])
        d_es_act = np.sqrt((ex - np.asarray(x_act))**2
                           + (ey - np.asarray(y_act))**2)
        fx = np.array(paired[seed]["full"]["x"])
        fy = np.array(paired[seed]["full"]["y"])
        d_full_act = np.sqrt((fx - np.asarray(x_act))**2
                             + (fy - np.asarray(y_act))**2)

        row = {
            "seed": seed,
            "windows": wins,
            "endpoint_percentile_in_tail_band": {
                "full": round(pct_full, 1), "es": round(pct_es, 1)},
            "es_final_vs_activation_disp_m": {
                "median": round(float(np.median(d_es_act)), 2),
                "max": round(float(d_es_act.max()), 2),
            },
            "full_final_vs_activation_disp_m": {
                "median": round(float(np.median(d_full_act)), 2),
                "max": round(float(d_full_act.max()), 2),
            },
        }
        out_rows.append(row)
        log(f"[seed {seed}] full@{pct_full:.0f}pct es@{pct_es:.0f}pct of band; "
            f"ES moved median {row['es_final_vs_activation_disp_m']['median']}m "
            f"(max {row['es_final_vs_activation_disp_m']['max']}m) from "
            f"activation; full moved median "
            f"{row['full_final_vs_activation_disp_m']['median']}m "
            f"(max {row['full_final_vs_activation_disp_m']['max']}m)")
        for w in row["windows"]:
            log(f"   steps {w['window']} lr~{w['lr_mid']:<6} mean "
                f"{w['band_mean']} slow_std {w['band_std_slow']} "
                f"fast_std {w['step_diff_std_fast']} p2p {w['band_p2p']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"seeds": out_rows}, f, indent=1)
    log(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
