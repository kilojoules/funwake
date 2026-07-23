#!/usr/bin/env python
"""Equivalent-cost SGD frontier: AEP vs number of multistarts K.

At a fixed budget of 8000 gradient evaluations, K vmapped multistarts of
T = 8000//K iterations each. Per-config mean +/- std with faint per-seed
scatter, against the 500-multistart baseline (~375x the eval cost) and the
schedule-mode seed at the same budget (infeasible).

Reads per-run JSONs from results/equiv_cost_sgd/frontier/ (written by
tools/run_budgeted_baseline.py).

Usage (from repo root):
    pixi run python scripts/plot_equiv_cost_frontier.py \
        [--out-base results/equiv_cost_sgd/fig_frontier]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRONTIER_DIR = "results/equiv_cost_sgd/frontier"

TRAIN_BASELINE = 5540.7      # best of 500 multistarts x 6000 iters (~3M evals)
BASELINE_COST_RATIO = 375    # ~3M evals / 8000 evals
SEED_SCHEDULE = 5529.2       # schedule-mode seed, same 8000-eval budget
NOISE_FLOOR = 0.3            # GWh; 10-seed config-mean indistinguishability

DATA_COLOR = "#1f77b4"       # single categorical hue (matplotlib C0)
REF_COLOR = "#555555"        # neutral reference: feasible baseline
SEED_COLOR = "#999999"       # neutral reference: infeasible seed schedule


def load_frontier(frontier_dir):
    """Group dei_K{K}_T{T}_seed{S}.json runs by K; return sorted stats."""
    pat = re.compile(r"dei_K(\d+)_T(\d+)_seed(\d+)\.json$")
    groups = {}
    for path in sorted(glob.glob(os.path.join(frontier_dir, "dei_K*_seed*.json"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        K = int(m.group(1))
        with open(path) as f:
            run = json.load(f)
        groups.setdefault(K, []).append(run)

    stats = []
    for K in sorted(groups):
        runs = groups[K]
        aeps = np.array([r["aep_gwh"] for r in runs])
        n_feas = sum(r["feasible"] for r in runs)
        stats.append({
            "K": K,
            "T": runs[0]["iters"],
            "aeps": aeps,
            "mean": aeps.mean(),
            "std": aeps.std(ddof=1),
            "n": len(runs),
            "n_feasible_runs": n_feas,
        })
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frontier-dir", default=FRONTIER_DIR)
    p.add_argument("--out-base", default="results/equiv_cost_sgd/fig_frontier",
                   help="Saves <out-base>.png and <out-base>.pdf")
    args = p.parse_args()

    stats = load_frontier(args.frontier_dir)
    if not stats:
        raise SystemExit(f"No frontier runs found in {args.frontier_dir}")

    Ks = np.array([s["K"] for s in stats])
    means = np.array([s["mean"] for s in stats])
    stds = np.array([s["std"] for s in stats])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Reference lines (neutral grays; identity carried by the legend text)
    ax.axhline(TRAIN_BASELINE, color=REF_COLOR, ls="--", lw=1.2, zorder=1,
               label=f"500-multistart baseline ({TRAIN_BASELINE:.1f}, "
                     f"~{BASELINE_COST_RATIO}x eval cost)")
    ax.axhline(SEED_SCHEDULE, color=SEED_COLOR, ls=(0, (2, 2)), lw=1.2, zorder=1,
               label=f"Schedule-mode seed, same budget "
                     f"({SEED_SCHEDULE:.1f}, INFEASIBLE)")

    # Faint per-seed scatter (slight multiplicative jitter for the log axis)
    rng = np.random.default_rng(0)
    for s in stats:
        jitter = s["K"] * np.exp(rng.uniform(-0.035, 0.035, size=s["n"]))
        ax.scatter(jitter, s["aeps"], color=DATA_COLOR, alpha=0.25, s=14,
                   linewidths=0, zorder=2)

    # Mean +/- std
    ax.errorbar(Ks, means, yerr=stds, color=DATA_COLOR, marker="o", ms=6,
                lw=1.8, capsize=3, capthick=1.2, elinewidth=1.2, zorder=4,
                label="Budgeted SGD, mean $\\pm$ std (10 seeds)")

    # K=1 feasibility annotation
    s1 = next((s for s in stats if s["K"] == 1), None)
    if s1 is not None:
        ax.annotate(f"{s1['n_feasible_runs']}/{s1['n']} feasible",
                    xy=(s1["K"], s1["mean"] - s1["std"]),
                    xytext=(s1["K"] * 1.12, s1["mean"] - s1["std"] - 3.2),
                    fontsize=8, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.7))

    # Plateau annotation (no turnover found through K=32)
    ax.annotate("plateau: K = 8$-$32 statistically tied\n"
                f"($\\pm${NOISE_FLOOR} GWh noise floor) $-$ "
                "no turnover through K = 32",
                xy=(9, 5536.4), fontsize=8, color="dimgray",
                ha="center", va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.minorticks_off()
    ax.set_xlabel("Multistarts $K$  ($T = 8000/K$ iterations each)")
    ax.set_ylabel("Best-feasible AEP (GWh)")
    ax.set_title("Equivalent-cost SGD frontier (8000 gradient evaluations, DEI farm)")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=9)

    for ext in ("png", "pdf"):
        out = f"{args.out_base}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
