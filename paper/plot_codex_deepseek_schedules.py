"""Plot schedule_fn curves for codex + deepseek 3-seed sched best scripts.

Loads each schedule, evaluates over 8000 steps with the standard lr0=50
and a representative alpha0, and renders a 4-panel comparison
(lr, alpha (log), beta1, beta2). Baseline seed schedule overlaid in
dashed black.

Usage:
    pixi run python paper/plot_codex_deepseek_schedules.py
"""
import importlib.util
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOTAL_STEPS = 8000
LR0 = 50.0
ALPHA0 = 0.0002545  # the empirical value from playground/problem.json N=50


SCRIPTS = [
    # label, path, color, ls
    ("Baseline (seed)",      "results/seed_schedule.py",                      "0.2", "--"),
    ("Codex sched run 1",    "results/agent_top_schedules/codex_sched_run1.py", "C0", "-"),
    ("Codex sched run 2",    "results/agent_top_schedules/codex_sched_run2.py", "C0", "-"),
    ("Codex sched run 3",    "results/agent_top_schedules/codex_sched_run3.py", "C0", "-"),
    ("DeepSeek sched run 1", "results/agent_top_schedules/deepseek_sched_run1.py", "C1", "-"),
    ("DeepSeek sched run 2", "results/agent_top_schedules/deepseek_sched_run2.py", "C1", "-"),
    ("DeepSeek sched run 3", "results/agent_top_schedules/deepseek_sched_run3.py", "C1", "-"),
]


def load_schedule_fn(path):
    spec = importlib.util.spec_from_file_location("u", os.path.join(PROJECT_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def trace(schedule_fn):
    lrs, alphas, b1s, b2s = [], [], [], []
    for i in range(TOTAL_STEPS):
        lr, alpha, b1, b2 = schedule_fn(jnp.array(i), TOTAL_STEPS, LR0, ALPHA0)
        lrs.append(float(lr));   alphas.append(float(alpha))
        b1s.append(float(b1));   b2s.append(float(b2))
    return np.array(lrs), np.array(alphas), np.array(b1s), np.array(b2s)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axs = axs.ravel()

    steps = np.arange(TOTAL_STEPS)

    handles_seen = {}
    for label, path, color, ls in SCRIPTS:
        try:
            fn = load_schedule_fn(path)
            lr, alpha, b1, b2 = trace(fn)
        except Exception as e:
            print(f"[plot] FAIL {label}: {e}")
            continue

        # Group label so legend shows one per provider+baseline
        group = label.split(" sched")[0] if "sched" in label else label
        legend_label = group if group not in handles_seen else None
        handles_seen.setdefault(group, True)

        for ax, y, title, ylog in zip(
            axs,
            [lr, alpha, b1, b2],
            ["learning rate", "alpha (log)", "beta1", "beta2"],
            [False, True, False, False],
        ):
            ax.plot(steps, y, color=color, ls=ls, alpha=0.85,
                    lw=1.4 if "Baseline" in label else 1.0,
                    label=legend_label)
            ax.set_title(title, fontsize=10)
            if ylog:
                ax.set_yscale("log")

    for ax in axs[2:]:
        ax.set_xlabel("step")
    axs[0].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Best schedules per agent run (lr0={LR0}, alpha0={ALPHA0:.4g}, "
        f"{TOTAL_STEPS} Adam steps)", fontsize=11)
    fig.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "paper", "short_codex_figs")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "codex_deepseek_schedules.png")
    out_pdf = os.path.join(out_dir, "codex_deepseek_schedules.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")


if __name__ == "__main__":
    main()
