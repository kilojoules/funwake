"""Analyse δ-sweep + iter_192 ES-on.

For each headline cell, gather all δ baseline results (across new sweep + the
default 0.01 from matrix_fair.json) and iter_192 results (ES-off from
matrix_fair, ES-on from delta_sweep_and_es). Compute per-cell:

  - best-δ value (max feasible AEP mean over δ values)
  - best-δ baseline AEP ± multi-seed std
  - iter_192 ES-off AEP ± std
  - iter_192 ES-on AEP ± std
  - gap iter_192 vs best-δ baseline
  - clears 0.2 vs best-δ? (gap ≥ 0.2 AND > spread)
  - ES-on vs ES-off delta in AEP + feasibility (Experiment B)
  - lr_ratio trace summary (when ES fires)

Outputs:
  - delta_sweep.csv      — per-cell table
  - delta_sweep.{pdf,png} — δ-curve per cell with iter_192 line + 0.2 bar
  - es_companion.{pdf,png} — iter_192 ES-on vs off + lr_ratio trace
  - REPORT_DELTA_SWEEP_ES.md
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


def gather_per_cell(matrix_fair_path, delta_sweep_path):
    """Return cells[cell_path] = {delta_runs: {δ: [aep,...]}, iter192_es_off: [...], iter192_es_on: [...], trace: ...}."""
    fair = json.load(open(matrix_fair_path))
    extra = json.load(open(delta_sweep_path))

    cells = {}

    # From matrix_fair: decay_es_baseline = δ=0.01 default, iter_192 = ES_OFF
    for r in fair["runs"]:
        if "error" in r:
            continue
        cp = r["cell_path"]
        if cp not in cells:
            cells[cp] = {"delta_runs": {}, "iter192_es_off": [], "iter192_es_on": [], "trace": None}
        if r["schedule"] == "decay_es_baseline":
            cells[cp]["delta_runs"].setdefault(0.01, []).append(r)
        elif r["schedule"] == "claude_iter192":
            cells[cp]["iter192_es_off"].append(r)

    # From delta_sweep: more deltas + iter192_es_on
    for r in extra["runs"]:
        if "error" in r:
            continue
        cp = r["cell_path"]
        if cp not in cells:
            cells[cp] = {"delta_runs": {}, "iter192_es_off": [], "iter192_es_on": [], "trace": None}
        if r["task_kind"] == "delta_baseline":
            cells[cp]["delta_runs"].setdefault(r["delta"], []).append(r)
        elif r["task_kind"] == "iter192_es_on":
            cells[cp]["iter192_es_on"].append(r)
            if r.get("lr_ratio_trajectory") and cells[cp]["trace"] is None:
                cells[cp]["trace"] = r["lr_ratio_trajectory"]
    return cells


def stat_runs(runs):
    """mean ± std of aep, plus bp_final/feasibility info."""
    if not runs:
        return None
    aeps = np.array([r["aep_det_gwh"] for r in runs])
    bps = np.array([r["bp_final"] for r in runs])
    sps = np.array([r["sp_final"] for r in runs])
    return {
        "mean": float(aeps.mean()),
        "std": float(aeps.std(ddof=1)) if len(aeps) > 1 else 0.0,
        "n": len(aeps),
        "bp_max": float(bps.max()),
        "sp_max": float(sps.max()),
        "practical_feas_frac": float(((bps < 1e-2) & (sps < 1e-2)).mean()),
    }


def per_cell_summary(cell_path, cell):
    """Compute best-δ AEP and gap vs iter_192."""
    delta_stats = {d: stat_runs(rs) for d, rs in cell["delta_runs"].items()}
    # Best δ: maximize mean AEP across δ values
    valid = {d: s for d, s in delta_stats.items() if s is not None}
    if not valid:
        return None
    best_delta, best_stat = max(valid.items(), key=lambda kv: kv[1]["mean"])

    iter192_off = stat_runs(cell["iter192_es_off"])
    iter192_on = stat_runs(cell["iter192_es_on"])

    gap_pct = None
    spread_pct = None
    if iter192_off is not None and best_stat["mean"] > 0:
        gap_pct = 100.0 * (iter192_off["mean"] - best_stat["mean"]) / best_stat["mean"]
        spread_pct = 100.0 * iter192_off["std"] / best_stat["mean"]
    clears = (
        gap_pct is not None and spread_pct is not None
        and gap_pct >= 0.2 and gap_pct > spread_pct
    )
    # ES on vs off delta
    es_delta = None
    if iter192_on is not None and iter192_off is not None:
        es_delta = {
            "aep_on_minus_off_gwh": iter192_on["mean"] - iter192_off["mean"],
            "aep_on_minus_off_pct": 100.0 * (iter192_on["mean"] - iter192_off["mean"]) / iter192_off["mean"],
            "feas_on_minus_off": iter192_on["practical_feas_frac"] - iter192_off["practical_feas_frac"],
        }

    return {
        "cell_path": cell_path,
        "delta_stats": {d: s for d, s in delta_stats.items() if s is not None},
        "best_delta": best_delta,
        "best_delta_aep_mean": best_stat["mean"],
        "best_delta_aep_std": best_stat["std"],
        "iter192_es_off": iter192_off,
        "iter192_es_on": iter192_on,
        "gap_vs_best_delta_pct": gap_pct,
        "spread_pct": spread_pct,
        "clears_0.2_vs_best_delta": clears,
        "es_on_vs_off": es_delta,
        "trace": cell["trace"],
    }


def parse_cell_meta(cell_path):
    base = os.path.basename(cell_path).replace("problem_", "").replace(".json", "")
    farm = base.split("_")[0]
    n = int(base.split("_n")[1].split("_")[0])
    rose = base.split("rose")[1]
    return {"farm": farm, "n": n, "rose": rose, "label": base}


def make_delta_sweep_figure(summaries, out_pdf, out_png):
    """One subplot per cell: AEP vs δ, with iter_192 horizontal line + 0.2% bar."""
    cells = sorted(summaries.keys())
    n_cells = len(cells)
    ncols = 6
    nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.0 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else axes if n_cells > 1 else [axes]

    for idx, cp in enumerate(cells):
        s = summaries[cp]
        if s is None:
            continue
        meta = parse_cell_meta(cp)
        ax = axes_flat[idx]
        deltas = sorted(s["delta_stats"].keys())
        means = [s["delta_stats"][d]["mean"] for d in deltas]
        stds = [s["delta_stats"][d]["std"] for d in deltas]
        ax.errorbar(deltas, means, yerr=stds, marker="o", linestyle="-",
                    color="#1a73c7", linewidth=1.6, markersize=6, capsize=3,
                    label="δ-baseline")
        # Highlight best δ
        bd = s["best_delta"]
        ax.scatter([bd], [s["best_delta_aep_mean"]],
                   s=180, marker="*", color="gold", edgecolor="black", zorder=10,
                   label=f"best δ={bd}")
        # iter_192 ES-off
        if s["iter192_es_off"] is not None:
            ax.axhline(s["iter192_es_off"]["mean"], color="#c73a1a", linewidth=1.6,
                        linestyle="-", label=f"iter_192 ES-off")
            # 0.2% bar above best-δ
            bar = s["best_delta_aep_mean"] * 1.002
            ax.axhline(bar, color="gray", linewidth=0.8, linestyle=":",
                        label="best-δ + 0.2 %")
        ax.set_xscale("log")
        ax.set_title(f"{meta['label']}", fontsize=8)
        ax.set_xlabel("δ (gamma_min_factor)", fontsize=8)
        ax.set_ylabel("AEP (GWh)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")
    # Hide unused
    for idx in range(n_cells, len(axes_flat)):
        axes_flat[idx].axis("off")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=180)
    plt.close()


def make_es_companion_figure(summaries, out_pdf, out_png):
    """Two-row figure: top = iter_192 ES-on vs off AEP delta per cell (bars);
       bottom = lr_ratio trajectory of iter_192 with ES threshold + first-cross."""
    cells = sorted(summaries.keys())
    n_cells = len(cells)
    fig = plt.figure(figsize=(18, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    labels = []
    deltas_aep = []
    deltas_aep_err = []
    deltas_feas = []
    for cp in cells:
        s = summaries[cp]
        if s is None or s["es_on_vs_off"] is None:
            continue
        meta = parse_cell_meta(cp)
        labels.append(meta["label"].replace("rose", ""))
        deltas_aep.append(s["es_on_vs_off"]["aep_on_minus_off_pct"])
        deltas_feas.append(s["es_on_vs_off"]["feas_on_minus_off"])
        # error: combine on + off std
        if s["iter192_es_on"] and s["iter192_es_off"]:
            err = np.sqrt(s["iter192_es_on"]["std"]**2 + s["iter192_es_off"]["std"]**2) / s["iter192_es_off"]["mean"] * 100
            deltas_aep_err.append(err)
        else:
            deltas_aep_err.append(0.0)

    xs = np.arange(len(labels))
    width = 0.35
    ax1.bar(xs - width/2, deltas_aep, width, yerr=deltas_aep_err, capsize=2,
            color="#c73a1a", label="AEP delta (%, ES on − off)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=60, fontsize=7, ha="right")
    ax1.set_ylabel("AEP delta % (on − off)", fontsize=8)
    ax1.set_title("Experiment B — iter_192 with ES on vs ES off (multi-seed mean ± combined std)", fontsize=10)
    ax1.legend(fontsize=8)

    # Bottom: pick one representative cell's lr_ratio trace
    rep_cell = None
    for cp in cells:
        if summaries[cp] and summaries[cp]["trace"]:
            rep_cell = cp
            break
    if rep_cell:
        tr = summaries[rep_cell]["trace"]
        steps = tr["steps_sampled"]
        ratio = tr.get("lr_ratio_to_running_max", [])
        ratio_init = tr.get("lr_ratio_to_lr_init_step0", [])
        ax2.plot(steps, ratio_init, color="#1a73c7", linewidth=1.6,
                 label="lr_i / lr_init (impl's ES check)")
        ax2.plot(steps, ratio, color="#c73a1a", linewidth=1.2, linestyle="--",
                 label="lr_i / running-max (cleaner reference)")
        ax2.axhline(0.1, color="gray", linestyle=":", linewidth=0.8,
                    label="ES threshold = 0.1")
        if tr.get("es_first_cross_step") is not None:
            ax2.axvline(tr["es_first_cross_step"], color="green", linestyle="--",
                        linewidth=0.8,
                        label=f"first crossing at step {tr['es_first_cross_step']}")
        ax2.set_xlabel("Iteration", fontsize=8)
        ax2.set_ylabel("lr_ratio", fontsize=8)
        ax2.set_title(f"iter_192 lr_ratio trajectory ({parse_cell_meta(rep_cell)['label']})",
                       fontsize=10)
        ax2.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_csv(summaries, out_path):
    rows = []
    for cp in sorted(summaries.keys()):
        s = summaries[cp]
        if s is None:
            continue
        meta = parse_cell_meta(cp)
        row = {
            "cell": meta["label"],
            "farm": meta["farm"], "n": meta["n"], "rose": meta["rose"],
            "best_delta": s["best_delta"],
            "best_delta_aep_mean": s["best_delta_aep_mean"],
            "best_delta_aep_std": s["best_delta_aep_std"],
            "iter192_off_aep_mean": (s["iter192_es_off"] or {}).get("mean"),
            "iter192_off_aep_std": (s["iter192_es_off"] or {}).get("std"),
            "iter192_on_aep_mean": (s["iter192_es_on"] or {}).get("mean"),
            "iter192_on_aep_std": (s["iter192_es_on"] or {}).get("std"),
            "gap_vs_best_delta_pct": s["gap_vs_best_delta_pct"],
            "spread_pct": s["spread_pct"],
            "clears_0.2_vs_best_delta": s["clears_0.2_vs_best_delta"],
            "es_on_minus_off_aep_pct": (s["es_on_vs_off"] or {}).get("aep_on_minus_off_pct"),
            "es_on_minus_off_feas": (s["es_on_vs_off"] or {}).get("feas_on_minus_off"),
            "es_fires_iter_0_warmup": (s["trace"] or {}).get("fires_at_iter_0_warmup"),
            "es_first_cross_step": (s["trace"] or {}).get("es_first_cross_step"),
        }
        rows.append(row)
    keys = list(rows[0].keys())
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-fair", default="validation/stochastic_aep/matrix_fair.json")
    p.add_argument("--delta-sweep", default="validation/stochastic_aep/delta_sweep_and_es.json")
    p.add_argument("--out-csv", default="validation/stochastic_aep/delta_sweep.csv")
    p.add_argument("--delta-pdf", default="validation/stochastic_aep/delta_sweep_curves.pdf")
    p.add_argument("--delta-png", default="validation/stochastic_aep/delta_sweep_curves.png")
    p.add_argument("--es-pdf", default="validation/stochastic_aep/es_companion.pdf")
    p.add_argument("--es-png", default="validation/stochastic_aep/es_companion.png")
    args = p.parse_args()

    cells = gather_per_cell(args.matrix_fair, args.delta_sweep)
    summaries = {cp: per_cell_summary(cp, c) for cp, c in cells.items()}

    rows = write_csv(summaries, args.out_csv)
    make_delta_sweep_figure(summaries, args.delta_pdf, args.delta_png)
    make_es_companion_figure(summaries, args.es_pdf, args.es_png)
    n_clear = sum(r["clears_0.2_vs_best_delta"] for r in rows)
    print(f"Headline cells: {len(rows)}")
    print(f"Clears 0.2 % vs best-δ: {n_clear}/{len(rows)}")
    es_pct = [r["es_on_minus_off_aep_pct"] for r in rows if r["es_on_minus_off_aep_pct"] is not None]
    if es_pct:
        print(f"ES on − off AEP %: mean={np.mean(es_pct):+.3f}  std={np.std(es_pct, ddof=1):+.3f}")
    print(f"Wrote {args.out_csv}, {args.delta_pdf}, {args.es_pdf}")


if __name__ == "__main__":
    main()
