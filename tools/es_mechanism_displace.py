#!/usr/bin/env python
"""ES-mechanism experiment 3: displacement attribution.

From the paired ES/full final layouts (experiment 1):
  - per-turbine displacement |ES_final - full_final| for every seed,
    distribution stats, and rank correlation with distance-to-boundary;
  - hybrid-layout attribution on selected seeds: starting from the full-run
    layout, replace the top-k most-displaced turbines with their ES
    positions (k = 1, 2, 3, 5, 10, all movers > 1 m, all n) and evaluate raw
    AEP of each hybrid. Also the reverse direction (ES layout with top-k
    reverted to full positions). Localized vs diffuse attribution.

Usage:
    pixi run python tools/es_mechanism_displace.py \
        --paired results/equiv_cost_sgd/es_mechanism/paired_10seeds.json \
        --out results/equiv_cost_sgd/es_mechanism/displacement.json
"""
import argparse
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from probe_es_truncation import load_problem, parse_seeds, log

import numpy as np
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty
from es_mechanism_tail import signed_boundary_dist


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_layout(objective, x, y, boundary):
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    aep = float(-objective(xj, yj))
    pen = float(boundary_penalty(xj, yj, boundary))
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    return aep, pen, float(d.min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?",
                   default=os.path.join(REPO_ROOT, "results",
                                        "problem_dei_n50.json"))
    p.add_argument("--paired",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism",
                                        "paired_10seeds.json"))
    p.add_argument("--hybrid-seeds", default=None,
                   help="Seeds for hybrid attribution (default: 0,1 plus "
                        "largest-|delta| seed)")
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "results", "equiv_cost_sgd",
                                        "es_mechanism", "displacement.json"))
    args = p.parse_args()

    paired = json.load(open(args.paired))
    objective, boundary, n_target, min_spacing = load_problem(args.problem)

    if args.hybrid_seeds is not None:
        hybrid_seeds = set(parse_seeds(args.hybrid_seeds))
    else:
        hybrid_seeds = {0, 1, paired["summary"]["largest_abs_delta_seed"]}

    per_seed = []
    for rec in paired["seeds"]:
        seed = rec["seed"]
        fx = np.array(rec["full"]["x"])
        fy = np.array(rec["full"]["y"])
        ex = np.array(rec["es"]["x"])
        ey = np.array(rec["es"]["y"])
        delta_total = rec["delta_es_minus_full_gwh"]

        disp = np.sqrt((ex - fx)**2 + (ey - fy)**2)
        bdist_full = signed_boundary_dist(fx, fy, boundary)
        bdist_es = signed_boundary_dist(ex, ey, boundary)

        order = np.argsort(-disp)  # most displaced first
        n_movers_1m = int((disp > 1.0).sum())
        entry = {
            "seed": seed,
            "delta_total_gwh": delta_total,
            "displacement": {
                "max_m": round(float(disp.max()), 2),
                "median_m": round(float(np.median(disp)), 3),
                "n_over_1m": n_movers_1m,
                "n_over_10m": int((disp > 10.0).sum()),
                "n_over_100m": int((disp > 100.0).sum()),
                "n_over_500m": int((disp > 500.0).sum()),
                "per_turbine_m": np.round(disp, 3).tolist(),
            },
            "boundary_dist_full_m": np.round(bdist_full, 2).tolist(),
            "boundary_dist_es_m": np.round(bdist_es, 2).tolist(),
            "spearman_disp_vs_boundary_dist": round(
                spearman(disp, bdist_full), 3),
            "top5_turbines": [
                {"idx": int(i), "disp_m": round(float(disp[i]), 2),
                 "bdist_full_m": round(float(bdist_full[i]), 2),
                 "bdist_es_m": round(float(bdist_es[i]), 2)}
                for i in order[:5]],
        }

        if seed in hybrid_seeds:
            ks = sorted({1, 2, 3, 5, 10, max(n_movers_1m, 1), n_target})
            hybrids = []
            for k in ks:
                top = order[:k]
                # forward: full layout, top-k moved to ES positions
                hx, hy = fx.copy(), fy.copy()
                hx[top], hy[top] = ex[top], ey[top]
                aep_f, pen_f, ms_f = eval_layout(objective, hx, hy, boundary)
                # reverse: ES layout, top-k reverted to full positions
                rx, ry = ex.copy(), ey.copy()
                rx[top], ry[top] = fx[top], fy[top]
                aep_r, pen_r, ms_r = eval_layout(objective, rx, ry, boundary)
                hybrids.append({
                    "k": int(k),
                    "fwd_aep_gwh": round(aep_f, 4),
                    "fwd_delta_vs_full_gwh": round(
                        aep_f - rec["full"]["aep_gwh"], 4),
                    "fwd_frac_of_total_delta": (
                        round((aep_f - rec["full"]["aep_gwh"]) / delta_total,
                              3) if abs(delta_total) > 1e-9 else None),
                    "fwd_min_spacing_m": round(ms_f, 1),
                    "fwd_boundary_penalty": pen_f,
                    "rev_aep_gwh": round(aep_r, 4),
                    "rev_delta_vs_es_gwh": round(
                        aep_r - rec["es"]["aep_gwh"], 4),
                    "rev_min_spacing_m": round(ms_r, 1),
                })
            entry["hybrids"] = hybrids
            log(f"[seed {seed}] delta={delta_total:+.3f} movers>1m="
                f"{n_movers_1m} max_disp={disp.max():.1f}m "
                f"spearman={entry['spearman_disp_vs_boundary_dist']}")
            for h in hybrids:
                log(f"  k={h['k']:>2} fwd_delta={h['fwd_delta_vs_full_gwh']:+.4f} "
                    f"({h['fwd_frac_of_total_delta']}) "
                    f"rev_delta_vs_es={h['rev_delta_vs_es_gwh']:+.4f}")
        per_seed.append(entry)

    # Cross-seed summary
    all_sp = [e["spearman_disp_vs_boundary_dist"] for e in per_seed]
    summary = {
        "n_seeds": len(per_seed),
        "mean_n_over_1m": float(np.mean(
            [e["displacement"]["n_over_1m"] for e in per_seed])),
        "mean_n_over_100m": float(np.mean(
            [e["displacement"]["n_over_100m"] for e in per_seed])),
        "spearman_disp_vs_boundary_dist_per_seed": all_sp,
        "mean_spearman": round(float(np.nanmean(all_sp)), 3),
    }

    out = {"problem": args.problem, "paired_source": args.paired,
           "seeds": per_seed, "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    log(f"[done] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
