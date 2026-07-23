#!/usr/bin/env python
"""ES-mechanism experiment 4: texture-averaged (smoothed) AEP comparison.

The raw AEP objective is piecewise-constant with ~0.1 GWh discrete jumps
under cm-scale layout changes (wake-cone radius mask |cw| < 2*sigma in
pixwake/deficit/base.py switches a finite e^-2 edge deficit on/off). Any
single-point AEP evaluation therefore carries a deterministic "texture"
offset of std ~0.15-0.2 GWh relative to the locally-averaged landscape.

This script estimates the SMOOTH component of AEP for each layout by
averaging raw AEP over K random perturbations (uniform +-scale per
coordinate), which decorrelates the texture:
    smoothed_AEP ~ local mean of AEP, SE ~ texture_std/sqrt(K)

Applied to, per seed: full endpoint, ES endpoint, activation point
(step-4079 replay, bit-verified update law). Decides whether the measured
ES-full deltas reflect real (smooth-landscape) quality differences or are
draws of the evaluation texture.

Usage:
    pixi run python tools/es_mechanism_smoothed.py \
        --out results/equiv_cost_sgd/es_mechanism/smoothed_deltas.json
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
from es_mechanism_tail import T_ACT, canon_state, make_scan

import numpy as np
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import (SGDSettings, _init_sgd_state,
                               _compute_mid_bisection)


def smoothed_aep(objective, x, y, rng, k=128, scale=0.25, chunk=32):
    """Mean raw AEP over k layouts perturbed uniformly +-scale per coord.

    Returns (mean, std_of_samples, standard_error, raw_unperturbed).
    """
    f = jax.jit(jax.vmap(lambda xx, yy: -objective(xx, yy)))
    px = x[None] + rng.uniform(-scale, scale, size=(k,) + x.shape)
    py = y[None] + rng.uniform(-scale, scale, size=(k,) + y.shape)
    vals = []
    for i in range(0, k, chunk):
        vals.append(np.asarray(f(jnp.asarray(px[i:i + chunk]),
                                 jnp.asarray(py[i:i + chunk]))))
    vals = np.concatenate(vals)
    raw = float(-objective(jnp.asarray(x), jnp.asarray(y)))
    return (float(vals.mean()), float(vals.std()),
            float(vals.std() / np.sqrt(k)), raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--paired",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "paired_10seeds.json"))
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "smoothed_deltas.json"))
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
    rng = np.random.default_rng(12345)

    rows = []
    for seed in seeds:
        t0 = time.time()
        pr = paired[seed]
        fx, fy = np.array(pr["full"]["x"]), np.array(pr["full"]["y"])
        ex, ey = np.array(pr["es"]["x"]), np.array(pr["es"]["y"])
        # activation-point layout via bit-verified replay
        init_x, init_y = grid_subsample_init(seed, boundary, n_target,
                                             min_spacing)
        g0x, g0y = grad_obj_fn(init_x, init_y)
        state0 = canon_state(_init_sgd_state(init_x, init_y, g0x, g0y,
                                             settings))
        ax_, ay_, _, _, _ = run(init_x, init_y, state0, T_ACT)
        ax, ay = np.asarray(ax_), np.asarray(ay_)

        sm = {}
        for name, (x, y) in [("full", (fx, fy)), ("es", (ex, ey)),
                             ("act", (ax, ay))]:
            mean, std, se, raw = smoothed_aep(objective, x, y, rng,
                                              k=args.k, scale=args.scale)
            sm[name] = {"smoothed": round(mean, 4),
                        "texture_std": round(std, 4),
                        "se": round(se, 4),
                        "raw": round(raw, 4),
                        "texture_offset_raw_minus_smoothed":
                            round(raw - mean, 4)}
        row = {
            "seed": seed,
            **{f"{k_}_{m}": v for k_, d in sm.items() for m, v in d.items()},
            "raw_delta_es_minus_full": round(
                sm["es"]["raw"] - sm["full"]["raw"], 4),
            "smoothed_delta_es_minus_full": round(
                sm["es"]["smoothed"] - sm["full"]["smoothed"], 4),
            "smoothed_tail_drift_full_minus_act": round(
                sm["full"]["smoothed"] - sm["act"]["smoothed"], 4),
            "smoothed_cleanup_es_minus_act": round(
                sm["es"]["smoothed"] - sm["act"]["smoothed"], 4),
        }
        rows.append(row)
        log(f"[seed {seed}] raw_d={row['raw_delta_es_minus_full']:+.3f} "
            f"smooth_d={row['smoothed_delta_es_minus_full']:+.3f} "
            f"(SE ~{sm['es']['se']:.3f}) "
            f"smooth_drift={row['smoothed_tail_drift_full_minus_act']:+.3f} "
            f"smooth_cleanup={row['smoothed_cleanup_es_minus_act']:+.3f} "
            f"texture_std={sm['full']['texture_std']:.3f} "
            f"({time.time()-t0:.1f}s)")

    rd = np.array([r["raw_delta_es_minus_full"] for r in rows])
    sd = np.array([r["smoothed_delta_es_minus_full"] for r in rows])
    drift = np.array([r["smoothed_tail_drift_full_minus_act"] for r in rows])
    cleanup = np.array([r["smoothed_cleanup_es_minus_act"] for r in rows])
    tex = np.array([r["full_texture_std"] for r in rows])
    summary = {
        "k": args.k, "scale_m": args.scale,
        "raw_delta_mean": round(float(rd.mean()), 4),
        "raw_delta_std": round(float(rd.std()), 4),
        "smoothed_delta_mean": round(float(sd.mean()), 4),
        "smoothed_delta_std": round(float(sd.std()), 4),
        "smoothed_delta_n_positive": int((sd > 0).sum()),
        "typical_se": round(float(np.mean(
            [r["es_se"] for r in rows])), 4),
        "corr_raw_vs_smoothed_delta": round(
            float(np.corrcoef(rd, sd)[0, 1]), 3),
        "mean_texture_std": round(float(tex.mean()), 4),
        "smoothed_tail_drift_mean": round(float(drift.mean()), 4),
        "smoothed_tail_drift_std": round(float(drift.std()), 4),
        "smoothed_cleanup_mean": round(float(cleanup.mean()), 4),
        "smoothed_cleanup_std": round(float(cleanup.std()), 4),
    }
    out = {"problem": args.problem, "rows": rows, "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    log(f"[done] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
