#!/usr/bin/env python
"""Generate all figures for the FunWake paper.

Outputs PNG + PDF files in paper/figs/.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

FIGS = os.path.join(os.path.dirname(__file__), "figs")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# Model colors
COLORS = {
    "Gemini CLI": "#4285f4",
    "Claude Code": "#d97757",
    "Qwen 32B": "#a020a0",
    "Llama 70B": "#1a9988",
    "Baseline": "#666666",
    "Seed": "#aaaaaa",
}


def load_results():
    """Load all result JSONs."""
    data = {}
    data["baselines"] = json.load(open(os.path.join(RESULTS, "baselines_500start.json")))
    data["gen_curve"] = json.load(open(os.path.join(RESULTS, "generalization_curve.json")))
    data["gen_sched"] = json.load(open(os.path.join(RESULTS, "generalization_schedule.json")))
    data["summary"] = json.load(open(os.path.join(RESULTS, "all_models_summary.json")))
    return data


def fig1_best_rowp_comparison(data):
    """Bar chart: best ROWP AEP for each model vs baseline."""
    baseline_rowp = data["baselines"]["problem_rowp"]["best_aep"]

    models = [
        ("Gemini CLI", 4328.0),
        ("Claude Code", 4271.5),
        ("Llama 70B", 4252.0),
        ("Qwen 32B", 4251.4),
        ("Baseline\n(500-start)", baseline_rowp),
    ]
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    deltas = [s - baseline_rowp for s in scores]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [COLORS["Gemini CLI"], COLORS["Claude Code"], COLORS["Llama 70B"],
              COLORS["Qwen 32B"], COLORS["Baseline"]]
    bars = ax.bar(names, scores, color=colors, edgecolor="black", linewidth=0.7)

    ax.axhline(baseline_rowp, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(4200, 4360)
    ax.set_ylabel("Held-out AEP (GWh)")
    ax.set_title("Best held-out (ROWP) performance by model", fontweight="bold")

    # Annotate deltas
    for bar, delta, score in zip(bars, deltas, scores):
        height = bar.get_height()
        label = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight the winner
    bars[0].set_edgecolor("#1a5490")
    bars[0].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig1_rowp_comparison.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig1_rowp_comparison.pdf"))
    plt.close()
    print("  fig1_rowp_comparison.png")


def fig2_generalization_curve(data):
    """AEP vs n_target curve for each model."""
    baselines = data["baselines"]
    gen = data["gen_curve"]
    gen_sched = data["gen_sched"]

    ns = [30, 40, 50, 60, 70, 80]

    # Baseline: 500-start SGD
    baseline_aep = []
    for n in ns:
        key = f"problem_dei_n{n}"
        baseline_aep.append(baselines[key]["best_aep"])

    # Scripts from gen_curve
    scripts = {}
    for r in gen["data"] if isinstance(gen, dict) and "data" in gen else gen:
        if not isinstance(r, dict):
            continue
        s = r.get("script", "")
        n = r.get("n_target", 0)
        aep = r.get("aep_gwh")
        if aep is None:
            continue
        if s not in scripts:
            scripts[s] = {}
        scripts[s][n] = aep

    for r in gen_sched if isinstance(gen_sched, list) else []:
        s = r.get("script", "")
        n = r.get("n_target", 0)
        aep = r.get("aep_gwh")
        if aep is None:
            continue
        if s not in scripts:
            scripts[s] = {}
        scripts[s][n] = aep

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Reference: 500-start baseline
    ax.plot(ns, baseline_aep, "o-", color="black",
            label="500 multi-start SGD (baseline)",
            linewidth=2.2, markersize=7, zorder=5)

    # LLM scripts — pick the best known per-model
    plot_scripts = [
        ("results_agent_schedule_only_5hr/iter_192.py", "Claude schedule (iter 192)", COLORS["Claude Code"], "s"),
        ("results_agent_gemini_cli_5hr/iter_192.py", "Gemini schedule (iter 192)", COLORS["Gemini CLI"], "^"),
        ("results_agent_qwen2_5-coder-32b_s1/iter_011.py", "Qwen 32B full-opt", COLORS["Qwen 32B"], "D"),
        ("results_agent_llama3_3-70b_s2/iter_002.py", "Llama 70B full-opt", COLORS["Llama 70B"], "v"),
        ("results/seed_optimizer.py", "Seed (single-start)", COLORS["Seed"], "x"),
    ]

    for script_key, label, color, marker in plot_scripts:
        if script_key not in scripts:
            continue
        d = scripts[script_key]
        xs = [n for n in ns if n in d]
        ys = [d[n] for n in xs]
        ax.plot(xs, ys, marker=marker, linestyle="-", color=color,
                label=label, linewidth=1.4, markersize=6)

    ax.set_xlabel("Number of turbines")
    ax.set_ylabel("AEP (GWh)")
    ax.set_title("Generalization across turbine counts (DEI farm)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig2_generalization.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig2_generalization.pdf"))
    plt.close()
    print("  fig2_generalization.png")


def fig3_discovered_schedules(data):
    """Plot the 4 schedule parameters over time for key schedules."""
    import sys
    sys.path.insert(0, "playground/pixwake/src")
    sys.path.insert(0, "playground")
    sys.path.insert(0, "results")

    total_steps = 8000
    lr0 = 50.0
    alpha0 = 1.0  # proxy

    # Import schedules
    import importlib.util

    schedules = []

    # Seed
    spec = importlib.util.spec_from_file_location("seed", "results/seed_schedule.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schedules.append(("Seed (baseline)", mod.schedule_fn, COLORS["Seed"], "-"))

    # Claude iter_192
    spec = importlib.util.spec_from_file_location("claude", "results_agent_schedule_only_5hr/iter_192.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schedules.append(("Claude (iter 192)", mod.schedule_fn, COLORS["Claude Code"], "-"))

    # Gemini iter_067 (best ROWP)
    spec = importlib.util.spec_from_file_location("gemini", "results_agent_gemini_cli_5hr/iter_067.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schedules.append(("Gemini (iter 67)", mod.schedule_fn, COLORS["Gemini CLI"], "-"))

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
    steps = np.arange(total_steps)
    t = steps / total_steps

    labels = ["Learning rate (lr)", r"Penalty weight ($\alpha$)",
              r"Adam $\beta_1$", r"Adam $\beta_2$"]

    for name, fn, color, ls in schedules:
        lrs = np.zeros(total_steps)
        alphas = np.zeros(total_steps)
        b1s = np.zeros(total_steps)
        b2s = np.zeros(total_steps)
        for i in range(total_steps):
            try:
                lr, a, b1, b2 = fn(i, total_steps, lr0, alpha0)
                lrs[i] = float(lr)
                alphas[i] = float(a)
                b1s[i] = float(b1)
                b2s[i] = float(b2)
            except Exception:
                lrs[i] = alphas[i] = b1s[i] = b2s[i] = np.nan

        axes[0, 0].plot(t, lrs, label=name, color=color, linestyle=ls, linewidth=1.5)
        axes[0, 1].plot(t, alphas, label=name, color=color, linestyle=ls, linewidth=1.5)
        axes[1, 0].plot(t, b1s, label=name, color=color, linestyle=ls, linewidth=1.5)
        axes[1, 1].plot(t, b2s, label=name, color=color, linestyle=ls, linewidth=1.5)

    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Progress (t = step / total_steps)")
    axes[1, 1].set_xlabel("Progress (t = step / total_steps)")
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(loc="upper right", frameon=True)

    plt.suptitle("Discovered schedule structures", fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig3_schedules.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig3_schedules.pdf"))
    plt.close()
    print("  fig3_schedules.png")


def fig4_train_vs_rowp(data):
    """Scatter: train AEP vs ROWP AEP for all scored attempts from each model."""
    import os

    sources = [
        ("Claude Code (schedule)", "results_agent_schedule_only_5hr/attempt_log.json", COLORS["Claude Code"], "o"),
        ("Gemini CLI (schedule)", "results_agent_gemini_cli_5hr/attempt_log.json", COLORS["Gemini CLI"], "^"),
        ("Qwen 32B (full-opt)", "results_agent_qwen2_5-coder-32b_s1/attempt_log.json", COLORS["Qwen 32B"], "D"),
        ("Llama 70B (full-opt)", "results_agent_llama3_3-70b_s2/attempt_log.json", COLORS["Llama 70B"], "v"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for name, path, color, marker in sources:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        points = [(a["train_aep"], a["rowp_aep"])
                  for a in d if "train_aep" in a and "rowp_aep" in a]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color=color, marker=marker, s=22, alpha=0.55,
                   label=f"{name} ({len(points)})", edgecolors="none")

    # Baseline reference
    ax.axhline(data["baselines"]["problem_rowp"]["best_aep"],
               color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Baseline ROWP")
    ax.axvline(5540.7, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="Baseline train")

    ax.set_xlabel("Training AEP (GWh)")
    ax.set_ylabel("Held-out (ROWP) AEP (GWh)")
    ax.set_title("Training vs held-out AEP across all scored attempts", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig4_train_vs_rowp.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig4_train_vs_rowp.pdf"))
    plt.close()
    print("  fig4_train_vs_rowp.png")


def fig5_seed_reproducibility(data):
    """Bar chart: Claude 4-seed reproducibility."""
    seeds = [1, 2, 3, 4]
    best_trains = []
    for s in seeds:
        try:
            d = json.load(open(f"results_agent_claude_sched_s{s}/attempt_log.json"))
            scored = [a for a in d if "train_aep" in a]
            best_trains.append(max(a["train_aep"] for a in scored) if scored else 0)
        except FileNotFoundError:
            best_trains.append(0)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar([f"Seed {s}" for s in seeds], best_trains,
                  color=COLORS["Claude Code"], edgecolor="black", linewidth=0.7)
    ax.axhline(5540.7, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="500-start baseline")
    ax.axhline(5600.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
               label="Claude original (320 att)")
    ax.set_ylim(5520, 5620)
    ax.set_ylabel("Best training AEP (GWh)")
    ax.set_title("Claude Code schedule-only: seed reproducibility", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    for bar, v in zip(bars, best_trains):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig5_seed_reproducibility.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig5_seed_reproducibility.pdf"))
    plt.close()
    print("  fig5_seed_reproducibility.png")


def main():
    print("Generating figures...")
    data = load_results()
    fig1_best_rowp_comparison(data)
    fig2_generalization_curve(data)
    try:
        fig3_discovered_schedules(data)
    except Exception as e:
        print(f"  fig3 failed: {e}")
    fig4_train_vs_rowp(data)
    fig5_seed_reproducibility(data)
    print("Done. Figures in paper/figs/")


if __name__ == "__main__":
    main()
