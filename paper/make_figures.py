#!/usr/bin/env python
"""Generate all figures for the FunWake paper.

Uses FEASIBLE ROWP scores only — infeasible held-out layouts are
not counted regardless of their reported AEP.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

FIGS = os.path.join(os.path.dirname(__file__), "figs")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ROOT = os.path.join(os.path.dirname(__file__), "..")

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

COLORS = {
    "Claude Code": "#d97757",
    "Gemini CLI": "#4285f4",
    "Qwen 32B": "#a020a0",
    "Llama 70B": "#1a9988",
    "Baseline": "#666666",
    "Seed": "#aaaaaa",
    "Random": "#888888",
    "Bump DE": "#b35900",
}


MODEL_DIRS = {
    "Claude schedule (320att)": ("results_agent_schedule_only_5hr", "Claude Code"),
    "Claude 6hr":                ("results_agent_claude_6hr",       "Claude Code"),
    "Claude 7hr":                ("results_agent_claude_7hr",       "Claude Code"),
    "Gemini 5hr v2 (full-opt)":  ("results_agent_5hr_v2",           "Gemini CLI"),
    "Gemini 5hr v4 (full-opt)":  ("results_agent_5hr_v4",           "Gemini CLI"),
    "Gemini 5hr v6":             ("results_agent_5hr_v6",           "Gemini CLI"),
}


def load_feasible_leaderboard():
    """Return list of {dir, model, n_att, best_feas_rowp, best_feas_train, best_iter}."""
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results_agent_*"))):
        log = os.path.join(d, "attempt_log.json")
        if not os.path.exists(log):
            continue
        try:
            attempts = json.load(open(log))
        except Exception:
            continue
        # Identify model
        model = "?"
        h = os.path.join(d, "history.json")
        if os.path.exists(h):
            try:
                hd = json.load(open(h))
                m = hd.get("model", "")
                if "claude" in m.lower():
                    model = "Claude Code"
                elif "gemini" in m.lower():
                    model = "Gemini CLI"
            except Exception:
                pass
        if model == "?":
            base = os.path.basename(d)
            if "claude" in base or "schedule_only" in base:
                model = "Claude Code"
            elif "qwen" in base.lower():
                model = "Qwen 32B"
            elif "llama" in base.lower():
                model = "Llama 70B"
            elif "5hr_v" in base or "5hr_noinit" in base or "5hr" == base[-3:] or "30min_v" in base or "1hr_v" in base or base == "results_agent_schedule_5hr":
                model = "Gemini CLI"

        feas = [a for a in attempts if a.get("rowp_feasible") and "rowp_aep" in a]
        if not feas:
            continue
        best = max(feas, key=lambda a: a["rowp_aep"])
        out.append({
            "dir": d,
            "basename": os.path.basename(d),
            "model": model,
            "n_att": len(attempts),
            "n_feas": len(feas),
            "best_feas_rowp": best["rowp_aep"],
            "best_feas_train": best["train_aep"],
            "best_iter": best["attempt"],
            "script": os.path.join(d, f"iter_{best['attempt']:03d}.py"),
        })
    out.sort(key=lambda x: -x["best_feas_rowp"])
    return out


def fig1_best_rowp_comparison(data, leaderboard):
    """Best FEASIBLE ROWP AEP per model."""
    baseline = data["baselines"]["problem_rowp"]["best_aep"]
    by_model = {}
    for r in leaderboard:
        if r["model"] not in by_model or r["best_feas_rowp"] > by_model[r["model"]]["best_feas_rowp"]:
            by_model[r["model"]] = r

    models = sorted(by_model.items(), key=lambda kv: -kv[1]["best_feas_rowp"])
    names = [m[0] for m in models]
    scores = [m[1]["best_feas_rowp"] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [COLORS.get(n, "#999999") for n in names]
    bars = ax.bar(names + ["500-start SGD"], scores + [baseline],
                  color=colors + [COLORS["Baseline"]],
                  edgecolor="black", linewidth=0.7)

    ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(min(scores + [baseline]) - 5, max(scores + [baseline]) + 8)
    ax.set_ylabel("Best feasible held-out (ROWP) AEP (GWh)")
    ax.set_title("Best held-out performance (feasible only)", fontweight="bold")

    for bar, score in zip(bars, scores + [baseline]):
        delta = score - baseline
        label = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        if abs(delta) < 0.01:
            label = "baseline"
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.3,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars[0].set_edgecolor("#1a5490")
    bars[0].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig1_rowp_comparison.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig1_rowp_comparison.pdf"))
    plt.close()
    print("  fig1_rowp_comparison.png")


def fig2_generalization_curve(data):
    """AEP vs n_target for each model's best script."""
    baselines = data["baselines"]
    gen = data["gen_curve"]

    ns = [30, 40, 50, 60, 70, 80]

    baseline_aep = []
    for n in ns:
        key = f"problem_dei_n{n}"
        baseline_aep.append(baselines[key]["best_aep"])

    scripts = {}
    for r in (gen if isinstance(gen, list) else []):
        s = r.get("script", "")
        n = r.get("n_target", 0)
        aep = r.get("aep_gwh")
        if aep is None:
            continue
        scripts.setdefault(s, {})[n] = aep

    try:
        gen_sched = json.load(open(os.path.join(RESULTS, "generalization_schedule.json")))
        for r in gen_sched:
            s = r.get("script", "")
            n = r.get("n_target", 0)
            aep = r.get("aep_gwh")
            if aep is None:
                continue
            scripts.setdefault(s, {})[n] = aep
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.plot(ns, baseline_aep, "o-", color="black",
            label="500 multi-start SGD baseline",
            linewidth=2.2, markersize=7, zorder=5)

    plot_scripts = [
        ("results_agent_schedule_only_5hr/iter_192.py", "Claude schedule (iter 192)", COLORS["Claude Code"], "s"),
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
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle="-", color=color,
                    label=label, linewidth=1.4, markersize=6)

    ax.set_xlabel("Number of turbines")
    ax.set_ylabel("AEP (GWh)")
    ax.set_title("Generalization across turbine counts (DEI)", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig2_generalization.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig2_generalization.pdf"))
    plt.close()
    print("  fig2_generalization.png")


def fig3_top3_schedules(leaderboard):
    """Plot the top-3 feasible-ROWP schedule_fn scripts."""
    import importlib.util

    # Scan ALL schedule_fn scripts across all dirs, pick top 3 by feasible ROWP
    all_scheds = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results_agent_*"))):
        log = os.path.join(d, "attempt_log.json")
        if not os.path.exists(log):
            continue
        try:
            atts = json.load(open(log))
        except Exception:
            continue
        for a in atts:
            if "rowp_aep" not in a or not a.get("rowp_feasible"):
                continue
            script = os.path.join(d, f"iter_{a['attempt']:03d}.py")
            if not os.path.exists(script):
                continue
            try:
                code = open(script).read()
            except Exception:
                continue
            if "def schedule_fn" not in code:
                continue
            all_scheds.append({
                "script": script,
                "dir": os.path.basename(d),
                "iter": a["attempt"],
                "rowp": a["rowp_aep"],
                "train": a["train_aep"],
            })

    all_scheds.sort(key=lambda x: -x["rowp"])
    top = all_scheds[:3]
    if not top:
        print("  fig3: no schedule_fn scripts found")
        return

    total_steps = 8000
    lr0 = 50.0
    alpha0 = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)

    colors = [COLORS["Claude Code"], "#9c4a2f", "#6b3020"]
    # Also overlay baseline
    spec = importlib.util.spec_from_file_location("seed", os.path.join(ROOT, "results/seed_schedule.py"))
    seed_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_mod)
    top_all = [
        {"label": "Baseline seed", "fn": seed_mod.schedule_fn, "color": COLORS["Seed"], "ls": "--"},
    ]
    for i, r in enumerate(top):
        spec = importlib.util.spec_from_file_location(f"top{i}", r["script"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        label = f"#{i+1}: {r['dir']} iter {r['iter']} (ROWP {r['rowp']:.1f})"
        top_all.append({"label": label, "fn": mod.schedule_fn, "color": colors[i], "ls": "-"})

    steps = np.arange(total_steps)
    t = steps / total_steps
    labels = ["Learning rate", r"Penalty weight $\alpha$", r"Adam $\beta_1$", r"Adam $\beta_2$"]

    for entry in top_all:
        lrs = np.zeros(total_steps)
        alphas = np.zeros(total_steps)
        b1s = np.zeros(total_steps)
        b2s = np.zeros(total_steps)
        for i in range(total_steps):
            try:
                lr, a, b1, b2 = entry["fn"](i, total_steps, lr0, alpha0)
                lrs[i] = float(lr); alphas[i] = float(a)
                b1s[i] = float(b1); b2s[i] = float(b2)
            except Exception:
                lrs[i] = alphas[i] = b1s[i] = b2s[i] = np.nan

        for ax, y in zip(axes.flat, [lrs, alphas, b1s, b2s]):
            ax.plot(t, y, label=entry["label"], color=entry["color"],
                    linestyle=entry["ls"], linewidth=1.4)

    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Progress $t$")
    axes[1, 1].set_xlabel("Progress $t$")
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=True)

    plt.suptitle("Top-3 feasible schedule_fn discoveries", fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig3_top3_schedules.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig3_top3_schedules.pdf"))
    plt.close()
    print("  fig3_top3_schedules.png")


def fig4_running_best_with_deploy(leaderboard):
    """Running-best train and ROWP vs wall-clock time, with deployed marker.

    Mirrors the README comparison plot: top panel = best-so-far training,
    bottom panel = rowp of the currently-best-train script ("would-be deployed").
    Marks the final deployed script with a star.
    """
    # Which dirs to plot — one best per model
    model_dirs = {}
    for r in leaderboard:
        if r["model"] not in model_dirs:
            model_dirs[r["model"]] = r["dir"]

    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    deploy_points = []

    for model, d in model_dirs.items():
        log = os.path.join(d, "attempt_log.json")
        if not os.path.exists(log):
            continue
        try:
            atts = json.load(open(log))
        except Exception:
            continue
        # Sort by timestamp
        scored = [a for a in atts if "train_aep" in a]
        scored.sort(key=lambda a: a.get("timestamp", 0))
        if not scored:
            continue
        t0 = scored[0].get("timestamp", 0)
        ts_min = [(a.get("timestamp", 0) - t0) / 60 for a in scored]

        best_train_so_far = []
        deploy_rowp_so_far = []   # rowp of current best-train script
        best_t = -float("inf")
        deploy_r = None
        for a in scored:
            if a["train_aep"] > best_t:
                best_t = a["train_aep"]
                if a.get("rowp_feasible"):
                    deploy_r = a.get("rowp_aep")
                else:
                    deploy_r = None  # infeasible; don't deploy
            best_train_so_far.append(best_t)
            deploy_rowp_so_far.append(deploy_r if deploy_r is not None else np.nan)

        color = COLORS.get(model, "#999999")
        ax_t.plot(ts_min, best_train_so_far, "-", color=color,
                  label=f"{model} (n={len(scored)})", linewidth=1.6)
        ax_r.plot(ts_min, deploy_rowp_so_far, "-", color=color,
                  label=f"{model}", linewidth=1.6)

        # Mark deployed point (last non-nan deploy_rowp)
        valid = [(t, r) for t, r in zip(ts_min, deploy_rowp_so_far) if not np.isnan(r)]
        if valid:
            deploy_points.append((valid[-1][0], valid[-1][1], color, model))

    # Baseline lines
    train_base = 5540.7
    rowp_base = 4243.75
    ax_t.axhline(train_base, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
                 label=f"500-start baseline ({train_base:.0f})")
    ax_r.axhline(rowp_base, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
                 label=f"500-start baseline ({rowp_base:.0f})")

    # Mark deployed scripts
    for t, r, color, model in deploy_points:
        ax_r.plot(t, r, marker="*", color=color, markersize=16,
                  markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    ax_t.set_ylabel("Best-so-far\ntraining AEP (GWh)")
    ax_t.set_title("Running-best AEP over time", fontweight="bold")
    ax_t.legend(loc="lower right", frameon=True, fontsize=8)
    ax_t.grid(True, alpha=0.3)

    ax_r.set_ylabel("Held-out (ROWP) AEP\nof currently-deployed script (GWh)")
    ax_r.set_xlabel("Wall-clock time (minutes)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="lower right", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig4_running_best.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig4_running_best.pdf"))
    plt.close()
    print("  fig4_running_best.png")


def fig5_convergence(data):
    """Best-so-far ROWP vs attempt for LLMs + random + DE when available."""
    baseline = data["baselines"]["problem_rowp"]["best_aep"]

    sources = [
        ("Claude Code", "results_agent_schedule_only_5hr/attempt_log.json", COLORS["Claude Code"]),
        ("Gemini CLI", "results_agent_5hr_v2/attempt_log.json", COLORS["Gemini CLI"]),
        ("Random search", "results_random_search_320/attempt_log.json", COLORS["Random"]),
        ("Bump DE", "results_bump_opt/bump_opt_log.json", COLORS["Bump DE"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for name, rel_path, color in sources:
        path = os.path.join(ROOT, rel_path)
        if not os.path.exists(path):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if isinstance(d, dict) and "history" in d:
            d = d["history"]
        # Use ONLY feasible ROWP
        feas_pts = []
        for a in d:
            if "rowp_aep" not in a:
                continue
            if not a.get("rowp_feasible"):
                continue
            n = a.get("attempt", a.get("eval", len(feas_pts) + 1))
            feas_pts.append((n, a["rowp_aep"]))
        if not feas_pts:
            continue
        feas_pts.sort()
        xs = [p[0] for p in feas_pts]
        best = [feas_pts[0][1]]
        for _, y in feas_pts[1:]:
            best.append(max(best[-1], y))
        ax.plot(xs, best, "-", color=color, label=f"{name} (n={len(feas_pts)})", linewidth=1.6)

    ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"500-start baseline ({baseline:.0f})")
    ax.set_xlabel("Attempt number")
    ax.set_ylabel("Best-so-far feasible ROWP (GWh)")
    ax.set_title("Search convergence: LLM vs random/DE baselines",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig5_convergence.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig5_convergence.pdf"))
    plt.close()
    print("  fig5_convergence.png")


def fig6_alpha_mechanism(data):
    """Terminal alpha vs ROWP AEP across all schedule_fn scripts."""
    import importlib.util
    import warnings
    warnings.filterwarnings("ignore")

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    # Sources: any dir with schedule_fn scripts + attempt logs with rowp
    source_dirs = [
        ("Claude Code", "results_agent_schedule_only_5hr", COLORS["Claude Code"], "o"),
    ]
    for s in [1, 2, 3, 4]:
        source_dirs.append((f"Claude seed {s}", f"results_agent_claude_sched_s{s}", COLORS["Claude Code"], "o"))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    total_steps = 8000
    lr0 = 50.0
    alpha0 = 1.0

    any_plotted = False
    plotted_labels = set()
    for name, dir_path, color, marker in source_dirs:
        log = os.path.join(ROOT, dir_path, "attempt_log.json")
        if not os.path.exists(log):
            continue
        try:
            atts = json.load(open(log))
        except Exception:
            continue
        xs, ys = [], []
        for a in atts:
            if "rowp_aep" not in a or not a.get("rowp_feasible"):
                continue
            script = os.path.join(ROOT, dir_path, f"iter_{a['attempt']:03d}.py")
            if not os.path.exists(script):
                continue
            try:
                spec = importlib.util.spec_from_file_location("sched_mod", script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "schedule_fn"):
                    continue
                _, a_final, _, _ = mod.schedule_fn(total_steps - 1, total_steps, lr0, alpha0)
                a_final = float(a_final)
                if a_final <= 0 or not np.isfinite(a_final):
                    continue
                xs.append(a_final)
                ys.append(a["rowp_aep"])
            except Exception:
                continue
        if not xs:
            continue
        label = "Claude Code" if "Claude" in name else name
        if label in plotted_labels:
            label = None
        else:
            plotted_labels.add(label)
        ax.scatter(xs, ys, c=color, marker=marker, s=22, alpha=0.5,
                   label=label, edgecolors="none")
        any_plotted = True

    if not any_plotted:
        print("  fig6: no data")
        return

    baseline = data["baselines"]["problem_rowp"]["best_aep"]
    ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"500-start baseline ({baseline:.0f})")

    ax.set_xscale("log")
    ax.set_xlabel(r"Terminal constraint weight $\alpha(t=1)$")
    ax.set_ylabel("Feasible held-out (ROWP) AEP (GWh)")
    ax.set_title("Mechanism: high terminal penalty predicts generalization",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig6_alpha_mechanism.png"), dpi=200)
    plt.savefig(os.path.join(FIGS, "fig6_alpha_mechanism.pdf"))
    plt.close()
    print("  fig6_alpha_mechanism.png")


def load_results():
    data = {}
    data["baselines"] = json.load(open(os.path.join(RESULTS, "baselines_500start.json")))
    try:
        data["gen_curve"] = json.load(open(os.path.join(RESULTS, "generalization_curve.json")))
    except FileNotFoundError:
        data["gen_curve"] = []
    return data


def main():
    print("Generating figures...")
    data = load_results()
    leaderboard = load_feasible_leaderboard()
    print(f"  Feasible leaderboard: {len(leaderboard)} runs")
    print(f"  Top 3:")
    for r in leaderboard[:3]:
        print(f"    {r['basename']:<40s} {r['model']:<15s} ROWP={r['best_feas_rowp']:.1f}  train={r['best_feas_train']:.1f}")

    fig1_best_rowp_comparison(data, leaderboard)
    fig2_generalization_curve(data)
    try:
        fig3_top3_schedules(leaderboard)
    except Exception as e:
        print(f"  fig3 failed: {e}")
    try:
        fig4_running_best_with_deploy(leaderboard)
    except Exception as e:
        print(f"  fig4 failed: {e}")
    fig5_convergence(data)
    try:
        fig6_alpha_mechanism(data)
    except Exception as e:
        print(f"  fig6 failed: {e}")
    print("Done. Figures in paper/figs/")


if __name__ == "__main__":
    main()
