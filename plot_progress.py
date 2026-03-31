#!/usr/bin/env python
"""Plot agent progress: training AEP vs held-out (ROWP) AEP.

Each point represents one optimizer script the LLM submitted.
The same script is evaluated on both the training farm and the
held-out ROWP farm, so the points are paired.

Usage:
    python plot_progress.py results_agent_1hr_v7/attempt_log.json
    python plot_progress.py results_agent_1hr_v7/attempt_log.json --save out.png
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    with open(path) as f:
        return json.load(f)


def plot(log_path, baselines_path="results/baselines.json",
         rowp_baseline_path="results/baseline_rowp.json",
         train_farm="1", save_path=None):

    log = load_log(log_path)

    # Load baselines
    with open(baselines_path) as f:
        baselines = json.load(f)
    train_bl = baselines[train_farm]["aep_gwh"]

    rowp_bl = None
    try:
        with open(rowp_baseline_path) as f:
            rowp_bl = json.load(f)["aep_gwh"]
    except FileNotFoundError:
        pass

    # Extract successful attempts with both train and ROWP scores
    paired = [e for e in log if "train_aep" in e and "rowp_aep" in e]
    train_only = [e for e in log if "train_aep" in e and "rowp_aep" not in e]
    errors = [e for e in log if "error" in e]

    if not paired and not train_only:
        print("No successful attempts to plot.")
        return

    # Use elapsed time from first attempt as x-axis
    t0 = log[0]["timestamp"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # ── Top panel: Training AEP ──
    if paired:
        t_paired = [(e["timestamp"] - t0) / 60 for e in paired]
        train_aeps = [e["train_aep"] for e in paired]
        ax1.plot(t_paired, train_aeps, "o-", color="C0", ms=6, label="Training (DEI)")
    if train_only:
        t_train = [(e["timestamp"] - t0) / 60 for e in train_only]
        train_aeps_only = [e["train_aep"] for e in train_only]
        ax1.plot(t_train, train_aeps_only, "o", color="C0", ms=6, alpha=0.4,
                 label="Training (no ROWP)")

    ax1.axhline(train_bl, color="C0", ls="--", alpha=0.5,
                label=f"Baseline ({train_bl:.1f})")

    # Mark errors as x on the x-axis
    if errors:
        t_err = [(e["timestamp"] - t0) / 60 for e in errors]
        ax1.plot(t_err, [train_bl] * len(t_err), "x", color="red", ms=5,
                 alpha=0.4, label=f"Errors ({len(errors)})")

    ax1.set_ylabel("Training AEP (GWh)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("Agent Progress")

    # ── Bottom panel: ROWP AEP ──
    if paired:
        rowp_aeps = [e["rowp_aep"] for e in paired]
        feasible = [e.get("rowp_feasible", False) for e in paired]

        # Plot feasible and infeasible with different markers
        t_feas = [t for t, f in zip(t_paired, feasible) if f]
        r_feas = [r for r, f in zip(rowp_aeps, feasible) if f]
        t_infeas = [t for t, f in zip(t_paired, feasible) if not f]
        r_infeas = [r for r, f in zip(rowp_aeps, feasible) if not f]

        if t_feas:
            ax2.plot(t_feas, r_feas, "s-", color="C1", ms=6,
                     label="ROWP (feasible)")
        if t_infeas:
            ax2.plot(t_infeas, r_infeas, "s", color="C1", ms=6, alpha=0.3,
                     mfc="none", label="ROWP (infeasible)")

    if rowp_bl is not None:
        ax2.axhline(rowp_bl, color="C1", ls="--", alpha=0.5,
                    label=f"Baseline ({rowp_bl:.1f})")

    ax2.set_ylabel("ROWP AEP (GWh)")
    ax2.set_xlabel("Time (minutes)")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        out = str(log_path).replace("attempt_log.json", "progress.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot agent progress")
    p.add_argument("log", help="Path to attempt_log.json")
    p.add_argument("--baselines", default="results/baselines.json")
    p.add_argument("--rowp-baseline", default="results/baseline_rowp.json")
    p.add_argument("--train-farm", default="1")
    p.add_argument("--save", default=None, help="Output path (default: progress.png next to log)")
    args = p.parse_args()
    plot(args.log, args.baselines, args.rowp_baseline, args.train_farm, args.save)


if __name__ == "__main__":
    main()
