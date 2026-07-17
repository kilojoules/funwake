"""Analyse the 324-run hardening pass.

H1 — refined δ on 4 low-margin cells: confirm iter_192 still beats best-δ
H2 — fixed-ES iter_192 on 18 cells: confirm internalization holds w/o spurious trigger
H3 — multi-init iter_192 on 4 low-margin cells: confirm gap survives across inits
H4 — smart-start init on 18 cells × 3 schedules: confirm gap holds under fair init

Outputs CSV + console summary + figures.
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = "/Users/julianquick/portfolio_copy/funwake"


HEADLINE_CELLS = [
    f"results/matrix/problem_{farm}_n{n}_rose{rose}.json"
    for farm in ("dei", "rowp")
    for n in (60, 70, 80)
    for rose in ("omnidir", "dei", "rowp")
]
LOW_MARGIN_CELLS = [
    "results/matrix/problem_dei_n60_roserowp.json",
    "results/matrix/problem_rowp_n70_roseomnidir.json",
    "results/matrix/problem_rowp_n60_rosedei.json",
    "results/matrix/problem_rowp_n60_roserowp.json",
]


def stat(aeps):
    a = np.asarray(aeps)
    if len(a) == 0:
        return None
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


LR_INIT = 50.0  # η₀ in metres. ηT = LR_INIT × δ (Quick 2023 Eq. 13).


def eta_t_of(delta):
    """Return ηT (m) for a given δ in our setup. δ = ηT / η₀, η₀ = 50 m."""
    return LR_INIT * float(delta)


def gather():
    hard = json.load(open("validation/stochastic_aep/hardening.json"))
    fair = json.load(open("validation/stochastic_aep/matrix_fair.json"))
    sweep = json.load(open("validation/stochastic_aep/delta_sweep_and_es.json"))
    # Optional paper-exact point
    paper_point_path = "validation/stochastic_aep/eta_t_paper_point.json"
    if os.path.exists(paper_point_path):
        paper_point = json.load(open(paper_point_path))
    else:
        paper_point = {"runs": []}

    # Index helpers
    runs = {"h1": {}, "h2": {}, "h3": {}, "h4": {}}
    # H1: cell → delta → [aeps] (also fold in paper-exact-ηT point)
    for r in hard["runs"]:
        if "error" in r:
            continue
        if r["kind"] == "h1_refined_delta":
            runs["h1"].setdefault(r["cell_path"], {}).setdefault(r["delta"], []).append(r["aep_det_gwh"])
        elif r["kind"] == "h2_fixed_es_iter192":
            runs["h2"].setdefault(r["cell_path"], []).append(r)
        elif r["kind"] == "h3_multi_init":
            runs["h3"].setdefault(r["cell_path"], {}).setdefault(r["init_seed"], []).append(r["aep_det_gwh"])
        elif r["kind"] == "h4_smart_start":
            runs["h4"].setdefault(r["cell_path"], {}).setdefault(r["schedule"], []).append(r)
    for r in paper_point["runs"]:
        if "error" in r:
            continue
        runs["h1"].setdefault(r["cell_path"], {}).setdefault(r["delta"], []).append(r["aep_det_gwh"])

    # Existing data from matrix_fair: existing δ per cell + iter_192 ES-off (init_seed=0)
    existing_delta = {}  # cell → δ → [aeps]
    iter192_off = {}     # cell → [aeps]  (init_seed=0)
    iter192_off_runs = {}  # cell → [run records]
    decay_es_default = {}  # cell → [aeps]
    claude_default = {}    # cell → [aeps]
    gemini_default = {}    # cell → [aeps]
    for r in fair["runs"]:
        if "error" in r:
            continue
        cp = r["cell_path"]
        if r["schedule"] == "claude_iter192":
            iter192_off.setdefault(cp, []).append(r["aep_det_gwh"])
            iter192_off_runs.setdefault(cp, []).append(r)
            claude_default.setdefault(cp, []).append(r["aep_det_gwh"])
        elif r["schedule"] == "decay_es_baseline":
            existing_delta.setdefault(cp, {}).setdefault(0.01, []).append(r["aep_det_gwh"])
            decay_es_default.setdefault(cp, []).append(r["aep_det_gwh"])
        elif r["schedule"] == "gemini_iter192":
            gemini_default.setdefault(cp, []).append(r["aep_det_gwh"])

    # Existing δ-sweep + iter_192 ES-on (old trigger): from delta_sweep_and_es
    iter192_on_old_trigger = {}  # cell → [aeps]
    for r in sweep["runs"]:
        if "error" in r:
            continue
        if r["task_kind"] == "delta_baseline":
            cp = r["cell_path"]
            existing_delta.setdefault(cp, {}).setdefault(r["delta"], []).append(r["aep_det_gwh"])
        elif r["task_kind"] == "iter192_es_on":
            iter192_on_old_trigger.setdefault(r["cell_path"], []).append(r["aep_det_gwh"])

    return {
        "h1": runs["h1"],
        "h2": runs["h2"],
        "h3": runs["h3"],
        "h4": runs["h4"],
        "existing_delta": existing_delta,
        "iter192_off": iter192_off,
        "iter192_off_runs": iter192_off_runs,
        "iter192_on_old": iter192_on_old_trigger,
        "decay_es_default": decay_es_default,
        "claude_default": claude_default,
        "gemini_default": gemini_default,
    }


def cell_label(cp):
    base = os.path.basename(cp).replace("problem_", "").replace(".json", "")
    return base


def analyze_h1(data):
    """For each low-margin cell, combine H1 refined δ with existing δ. Find new best-δ.
    Compute iter_192 gap vs new best-δ. Verify still clears 0.2 %."""
    out = []
    for cp in LOW_MARGIN_CELLS:
        all_delta = dict(data["existing_delta"].get(cp, {}))
        for d, aeps in data["h1"].get(cp, {}).items():
            all_delta.setdefault(d, []).extend(aeps)
        delta_stats = {d: stat(a) for d, a in all_delta.items()}
        valid = {d: s for d, s in delta_stats.items() if s and s["n"] >= 1}
        if not valid:
            continue
        best_d, best_s = max(valid.items(), key=lambda kv: kv[1]["mean"])
        iter192 = stat(data["iter192_off"].get(cp, []))
        gap_pct = 100.0 * (iter192["mean"] - best_s["mean"]) / best_s["mean"] if iter192 else None
        spread_pct = 100.0 * iter192["std"] / best_s["mean"] if iter192 else None
        clears = gap_pct is not None and spread_pct is not None and gap_pct >= 0.2 and gap_pct > spread_pct
        # Compute baseline AEP at the paper's recommended ηT = 0.1 m
        # (δ = 0.002) if present in the sweep
        paper_s = delta_stats.get(0.002)
        if paper_s and iter192:
            gap_paper_pct = 100.0 * (iter192["mean"] - paper_s["mean"]) / paper_s["mean"]
            spread_paper_pct = 100.0 * iter192["std"] / paper_s["mean"]
        else:
            gap_paper_pct = None
            spread_paper_pct = None
        out.append({
            "cell": cell_label(cp),
            "n_delta_pts_total": len(valid),
            "best_delta_new": best_d,
            "best_eta_t_new_m": eta_t_of(best_d),
            "best_delta_aep": best_s["mean"],
            "iter192_aep": iter192["mean"] if iter192 else None,
            "gap_vs_new_best_pct": gap_pct,
            "spread_pct": spread_pct,
            "clears_0.2_vs_new_best": clears,
            "paper_eta_t_baseline_aep": paper_s["mean"] if paper_s else None,
            "gap_vs_paper_eta_t_pct": gap_paper_pct,
            "spread_vs_paper_pct": spread_paper_pct,
            "delta_stats": {str(d): s for d, s in delta_stats.items()},
            "eta_t_stats": {str(eta_t_of(d)): s for d, s in delta_stats.items()},
        })
    return out


def analyze_h2(data):
    """For each headline cell, compare iter_192 ES-on (running-max trigger) to
    iter_192 ES-off. Verify ES still does nothing meaningful."""
    out = []
    for cp in HEADLINE_CELLS:
        off_aeps = data["iter192_off"].get(cp, [])
        # H2 uses run-max trigger
        on_runs = data["h2"].get(cp, [])
        on_aeps = [r["aep_det_gwh"] for r in on_runs]
        old_on_aeps = data["iter192_on_old"].get(cp, [])
        off_s = stat(off_aeps)
        on_s = stat(on_aeps)
        old_s = stat(old_on_aeps)
        if off_s and on_s:
            delta_pct = 100.0 * (on_s["mean"] - off_s["mean"]) / off_s["mean"]
            delta_gwh = on_s["mean"] - off_s["mean"]
        else:
            delta_pct = None
            delta_gwh = None
        # Combined std for delta
        if off_s and on_s and (off_s["n"] > 1 or on_s["n"] > 1):
            comb_std_pct = (
                100.0 * np.sqrt(on_s["std"]**2 + off_s["std"]**2) / off_s["mean"]
            )
        else:
            comb_std_pct = None
        out.append({
            "cell": cell_label(cp),
            "iter192_off_mean": off_s["mean"] if off_s else None,
            "iter192_off_std": off_s["std"] if off_s else None,
            "iter192_runmax_es_mean": on_s["mean"] if on_s else None,
            "iter192_runmax_es_std": on_s["std"] if on_s else None,
            "iter192_oldtrig_es_mean": old_s["mean"] if old_s else None,
            "delta_runmax_minus_off_pct": delta_pct,
            "delta_runmax_minus_off_gwh": delta_gwh,
            "combined_std_pct": comb_std_pct,
        })
    return out


def analyze_h3(data):
    """For 4 low-margin cells, compare across-init spread to across-sample-seed
    spread. Get a real multi-init gap estimate."""
    out = []
    for cp in LOW_MARGIN_CELLS:
        # Original: init_seed=0 in matrix_fair (iter_192)
        seed0_aeps = data["iter192_off"].get(cp, [])
        # H3: init_seed 1, 2 in hardening
        new_inits = data["h3"].get(cp, {})
        # Per-init mean
        per_init_means = {}
        per_init_means[0] = stat(seed0_aeps)
        for is_, aeps in new_inits.items():
            per_init_means[is_] = stat(aeps)
        # Combined across all inits
        all_aeps = list(seed0_aeps)
        for aeps in new_inits.values():
            all_aeps.extend(aeps)
        combined = stat(all_aeps)
        # Across-init spread (mean of mean-per-init)
        per_init_mean_values = [s["mean"] for s in per_init_means.values() if s]
        across_init_std = float(np.std(per_init_mean_values, ddof=1)) if len(per_init_mean_values) > 1 else 0.0
        out.append({
            "cell": cell_label(cp),
            "iter192_mean_all_inits": combined["mean"] if combined else None,
            "iter192_std_all_inits": combined["std"] if combined else None,
            "iter192_across_init_std": across_init_std,
            "per_init_summary": {str(k): v for k, v in per_init_means.items()},
        })
    return out


def analyze_h4(data):
    """For each headline cell, compare smart-start init vs wind-aware-grid init.
    For each schedule, what's the gap iter_192/gemini over decay+ES baseline
    when ALL three use smart-start init?"""
    out = []
    for cp in HEADLINE_CELLS:
        h4 = data["h4"].get(cp, {})
        base_aeps = [r["aep_det_gwh"] for r in h4.get("decay_es_baseline", [])]
        claude_aeps = [r["aep_det_gwh"] for r in h4.get("claude_iter192", [])]
        gemini_aeps = [r["aep_det_gwh"] for r in h4.get("gemini_iter192", [])]
        base_s = stat(base_aeps)
        claude_s = stat(claude_aeps)
        gemini_s = stat(gemini_aeps)
        if not (base_s and claude_s and gemini_s):
            continue
        claude_gap = 100.0 * (claude_s["mean"] - base_s["mean"]) / base_s["mean"]
        gemini_gap = 100.0 * (gemini_s["mean"] - base_s["mean"]) / base_s["mean"]
        claude_spread = 100.0 * claude_s["std"] / base_s["mean"]
        gemini_spread = 100.0 * gemini_s["std"] / base_s["mean"]
        # Reference: wind-aware-init gap (matrix_fair)
        wa_base = stat(data["decay_es_default"].get(cp, []))
        wa_claude = stat(data["claude_default"].get(cp, []))
        wa_gemini = stat(data["gemini_default"].get(cp, []))
        wa_claude_gap = (
            100.0 * (wa_claude["mean"] - wa_base["mean"]) / wa_base["mean"]
            if wa_base and wa_claude else None
        )
        wa_gemini_gap = (
            100.0 * (wa_gemini["mean"] - wa_base["mean"]) / wa_base["mean"]
            if wa_base and wa_gemini else None
        )
        out.append({
            "cell": cell_label(cp),
            "ss_baseline_aep_mean": base_s["mean"],
            "ss_baseline_aep_std": base_s["std"],
            "ss_claude_aep_mean": claude_s["mean"],
            "ss_claude_aep_std": claude_s["std"],
            "ss_gemini_aep_mean": gemini_s["mean"],
            "ss_gemini_aep_std": gemini_s["std"],
            "claude_gap_ss_init_pct": claude_gap,
            "claude_spread_ss_init_pct": claude_spread,
            "gemini_gap_ss_init_pct": gemini_gap,
            "gemini_spread_ss_init_pct": gemini_spread,
            "wa_claude_gap_pct": wa_claude_gap,
            "wa_gemini_gap_pct": wa_gemini_gap,
            "claude_clears_0.2_under_ss_init": claude_gap >= 0.2 and claude_gap > claude_spread,
            "gemini_clears_0.2_under_ss_init": gemini_gap >= 0.2 and gemini_gap > gemini_spread,
        })
    return out


def main():
    data = gather()
    h1 = analyze_h1(data)
    h2 = analyze_h2(data)
    h3 = analyze_h3(data)
    h4 = analyze_h4(data)

    # Save per-experiment CSVs
    for name, rows in [("h1_refined_delta", h1), ("h2_fixed_es", h2),
                        ("h3_multi_init", h3), ("h4_smart_start", h4)]:
        if not rows:
            continue
        # Flatten nested dict fields
        flat_rows = []
        for r in rows:
            fr = {k: v for k, v in r.items() if not isinstance(v, dict)}
            flat_rows.append(fr)
        with open(f"validation/stochastic_aep/{name}.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            for r in flat_rows:
                w.writerow(r)

    # Console summary
    print("=== H1 — refined ηT on 4 low-margin cells (also reported as δ) ===")
    for r in h1:
        f = "✓" if r["clears_0.2_vs_new_best"] else "✗"
        paper_str = (f"  paper-ηT(0.1m)_gap={r['gap_vs_paper_eta_t_pct']:+.3f}%"
                     if r["gap_vs_paper_eta_t_pct"] is not None else "  paper-ηT: not yet measured")
        print(f"  {f} {r['cell']:30s} best_ηT={r['best_eta_t_new_m']:>5.2f}m  "
              f"iter192_gap={r['gap_vs_new_best_pct']:+.3f}±{r['spread_pct']:.3f}%  "
              f"(pts={r['n_delta_pts_total']}){paper_str}")
    n1 = sum(r["clears_0.2_vs_new_best"] for r in h1)
    print(f"  → clears 0.2 % vs new best-ηT: {n1}/{len(h1)}")

    print("\n=== H2 — fixed-ES (running-max trigger) iter_192 on 18 cells ===")
    delta_pcts = []
    for r in h2:
        dp = r["delta_runmax_minus_off_pct"]
        if dp is not None:
            delta_pcts.append(dp)
            print(f"  {r['cell']:30s} ΔAEP_runmax-off={dp:+.4f}%  (oldtrig={r['iter192_oldtrig_es_mean']:.2f} vs runmax={r['iter192_runmax_es_mean']:.2f})")
    if delta_pcts:
        mean_d = float(np.mean(delta_pcts))
        std_d = float(np.std(delta_pcts, ddof=1))
        print(f"  → mean ΔAEP (ES on − off, running-max): {mean_d:+.4f} % ± {std_d:.4f}")

    print("\n=== H3 — multi-init iter_192 on 4 low-margin cells ===")
    for r in h3:
        print(f"  {r['cell']:30s} mean_all_inits={r['iter192_mean_all_inits']:.2f}  std_all={r['iter192_std_all_inits']:.3f}  across_init_std={r['iter192_across_init_std']:.3f}")

    print("\n=== H4 — smart-start init on 18 cells ===")
    cells_clear_h4 = 0
    for r in h4:
        cclr = "✓" if r["claude_clears_0.2_under_ss_init"] else "✗"
        gclr = "✓" if r["gemini_clears_0.2_under_ss_init"] else "✗"
        if r["claude_clears_0.2_under_ss_init"]:
            cells_clear_h4 += 1
        print(f"  C{cclr} G{gclr}  {r['cell']:30s}  C_gap={r['claude_gap_ss_init_pct']:+.3f}±{r['claude_spread_ss_init_pct']:.3f}%  (wa_init: C={r['wa_claude_gap_pct']:+.3f}%)")
    print(f"  → Claude clears 0.2 % under smart-start init: {cells_clear_h4}/{len(h4)}")

    # Save summary JSON for full traceability
    out = {"h1": h1, "h2": h2, "h3": h3, "h4": h4}
    with open("validation/stochastic_aep/hardening_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote h1/h2/h3/h4 CSVs + hardening_summary.json")


if __name__ == "__main__":
    main()
