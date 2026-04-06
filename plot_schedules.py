#!/usr/bin/env python
"""Plot scheduling curves: baseline vs LLM-discovered.

Shows lr, alpha, beta1, beta2 over optimization steps for:
- TopFarm baseline (coupled decay)
- LLM deployed (iter_192: dual bumps + alpha dip)

Usage:
    python plot_schedules.py --save schedules.png
"""
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_schedule(path):
    spec = importlib.util.spec_from_file_location("sched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def evaluate_schedule(schedule_fn, total_steps=8000, lr0=50.0, alpha0=10.0):
    """Evaluate schedule at every step, return arrays."""
    steps = np.arange(total_steps)
    lr = np.zeros(total_steps)
    alpha = np.zeros(total_steps)
    beta1 = np.zeros(total_steps)
    beta2 = np.zeros(total_steps)

    for i in steps:
        l, a, b1, b2 = schedule_fn(float(i), float(total_steps), lr0, alpha0)
        lr[i] = float(l)
        alpha[i] = float(a)
        beta1[i] = float(b1)
        beta2[i] = float(b2)

    t = steps / total_steps
    return t, lr, alpha, beta1, beta2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="schedules.png")
    args = p.parse_args()

    # Load schedules
    baseline_fn = load_schedule("results/seed_schedule.py")
    deployed_fn = load_schedule("results_agent_schedule_only_5hr/iter_192.py")

    t_b, lr_b, alpha_b, b1_b, b2_b = evaluate_schedule(baseline_fn)
    t_d, lr_d, alpha_d, b1_d, b2_d = evaluate_schedule(deployed_fn)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    # LR
    ax = axes[0]
    ax.plot(t_b, lr_b, "-", color="gray", linewidth=2, label="Baseline (TopFarm-style)")
    ax.plot(t_d, lr_d, "-", color="C3", linewidth=2, label="LLM deployed (iter 192)")
    ax.set_ylabel("Learning rate")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Optimizer Schedule: Baseline vs LLM-Discovered")
    # Annotate bumps
    ax.annotate("bump 1\n(t=0.5)", xy=(0.5, lr_d[int(0.5*len(lr_d))]),
                fontsize=8, color="C3", ha="center", va="bottom")
    ax.annotate("bump 2\n(t=0.75)", xy=(0.75, lr_d[int(0.75*len(lr_d))]),
                fontsize=8, color="C3", ha="center", va="bottom")

    # Alpha
    ax = axes[1]
    ax.plot(t_b, alpha_b, "-", color="gray", linewidth=2, label="Baseline")
    ax.plot(t_d, alpha_d, "-", color="C3", linewidth=2, label="LLM deployed")
    ax.set_ylabel("Penalty weight (α)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    # Annotate dip
    dip_idx = int(0.6 * len(alpha_d))
    ax.annotate("α dip\n(t=0.6)", xy=(0.6, alpha_d[dip_idx]),
                fontsize=8, color="C3", ha="center", va="top")

    # Beta1
    ax = axes[2]
    ax.plot(t_b, b1_b, "-", color="gray", linewidth=2, label="Baseline (0.1)")
    ax.plot(t_d, b1_d, "-", color="C3", linewidth=2, label="LLM deployed (0.3)")
    ax.set_ylabel("β₁ (momentum)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)

    # Beta2
    ax = axes[3]
    ax.plot(t_b, b2_b, "-", color="gray", linewidth=2, label="Baseline (0.2)")
    ax.plot(t_d, b2_d, "-", color="C3", linewidth=2, label="LLM deployed (0.5)")
    ax.set_ylabel("β₂ (adaptive)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Optimization progress (t = step / total_steps)")
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
