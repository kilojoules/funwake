"""Task A + B + C analysis.

A — gradient-balance discriminator: ratio r = ‖∇AEP‖ / (α·‖∇γ‖) for
    iter_192 vs baseline ES-off across iterations. Late-phase (iter 6400+):
    iter_192's r ≥ O(1) → productive-tail (AEP gradient still meaningful).
    iter_192's r collapses → dead-zone-avoidance (feasibility via other route).

B — baseline ES-cost ΔAEP = endpoint(ES-on) − endpoint(ES-off) per cell.
    Asymmetry: baseline ES-cost vs iter_192 ES-cost (from H2 or per-cell here).

C — two-curve mechanism figure: iter_192 ES-off/on + baseline ES-off/on per cell.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


TRACE_DIR = "/Users/julianquick/portfolio_copy/funwake/validation/stochastic_aep"
ITER_192_REAL_ES_TRIGGER = 6400  # where running-max trigger first fires
ITER_192_LR_PEAK = 200.0

CELLS = [
    ("rowp_n80_roserowp", 5.0),
    ("rowp_n80_roseomnidir", 1.0),
    ("rowp_n70_roserowp", 1.0),
]


def load(cell, kind):
    """kind = iter192_off | iter192_on | baseline_eta{eta}_off | baseline_eta{eta}_on"""
    p = os.path.join(TRACE_DIR, f"gb_{cell}_{kind}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def per_iter_means(d):
    """Average across sample seeds per iter. Returns arrays."""
    runs = [r for r in d["runs"] if "error" not in r]
    if not runs:
        return None
    iters = np.asarray([t["iter"] for t in runs[0]["trace"]])
    fields = ["aep", "bp", "sp", "grad_obj_norm", "grad_con_norm",
              "alpha", "lr", "alpha_grad_con_norm", "ratio_obj_over_alpha_con"]
    out = {f: np.stack([[t[f] for t in r["trace"]] for r in runs]).mean(0)
            for f in fields}
    out["iters"] = iters
    return out


def main():
    print("=" * 80)
    print("Task A — gradient-balance discriminator")
    print("=" * 80)
    rows = []
    for cell, eta_t in CELLS:
        iter192_off = per_iter_means(load(cell, "iter192_off"))
        baseline_off = per_iter_means(load(cell, f"baseline_eta{eta_t}_off"))
        if iter192_off is None or baseline_off is None:
            print(f"{cell}: missing data"); continue

        # Late phase: iter 6400-8000 (last 8 probe points at probe_every=200)
        iters = iter192_off["iters"]
        late_mask = iters >= ITER_192_REAL_ES_TRIGGER
        r192_late = iter192_off["ratio_obj_over_alpha_con"][late_mask]
        rb_late = baseline_off["ratio_obj_over_alpha_con"][late_mask]
        # Early phase for reference: iter 1000-3000 (mid optimization)
        early_mask = (iters >= 1000) & (iters <= 3000)
        r192_early = iter192_off["ratio_obj_over_alpha_con"][early_mask]
        rb_early = baseline_off["ratio_obj_over_alpha_con"][early_mask]

        print(f"\n{cell}:")
        print(f"  iter_192 ratio EARLY (iters 1000-3000): mean={r192_early.mean():.3f}, range=[{r192_early.min():.3f}, {r192_early.max():.3f}]")
        print(f"  iter_192 ratio LATE  (iters {ITER_192_REAL_ES_TRIGGER}-8000): mean={r192_late.mean():.3f}, range=[{r192_late.min():.3f}, {r192_late.max():.3f}]")
        print(f"  baseline ratio EARLY: mean={rb_early.mean():.3f}, range=[{rb_early.min():.3f}, {rb_early.max():.3f}]")
        print(f"  baseline ratio LATE:  mean={rb_late.mean():.3f}, range=[{rb_late.min():.3f}, {rb_late.max():.3f}]")
        late_ratio_ratio = r192_late.mean() / max(rb_late.mean(), 1e-30)
        print(f"  → iter_192_late_ratio / baseline_late_ratio = {late_ratio_ratio:.2f}×")
        # Pre-reg read
        if r192_late.mean() >= 0.5:  # O(1)
            verdict = "PRODUCTIVE-TAIL — iter_192's AEP gradient still competitive late"
        elif r192_late.mean() < 0.1 and rb_late.mean() < 0.1:
            verdict = "DEAD-ZONE-AVOIDANCE — both collapse late, iter_192 reaches feasibility some other way"
        else:
            verdict = f"MIXED — iter_192 late r={r192_late.mean():.3f}, baseline late r={rb_late.mean():.3f}"
        print(f"  → Verdict: {verdict}")
        rows.append({
            "cell": cell, "iter192_late_r": r192_late.mean(),
            "baseline_late_r": rb_late.mean(),
            "ratio_iter192_over_baseline_late": late_ratio_ratio,
            "verdict": verdict,
        })

    print("\n" + "=" * 80)
    print("Task B — baseline ES-cost asymmetry")
    print("=" * 80)
    asym_rows = []
    for cell, eta_t in CELLS:
        i_off = per_iter_means(load(cell, "iter192_off"))
        i_on = per_iter_means(load(cell, "iter192_on"))
        b_off = per_iter_means(load(cell, f"baseline_eta{eta_t}_off"))
        b_on = per_iter_means(load(cell, f"baseline_eta{eta_t}_on"))
        if not all([i_off, i_on, b_off, b_on]):
            print(f"{cell}: missing"); continue
        i_es_cost_abs = i_on["aep"][-1] - i_off["aep"][-1]
        i_es_cost_pct = 100 * i_es_cost_abs / i_off["aep"][-1]
        b_es_cost_abs = b_on["aep"][-1] - b_off["aep"][-1]
        b_es_cost_pct = 100 * b_es_cost_abs / b_off["aep"][-1]
        ratio = b_es_cost_pct / i_es_cost_pct if abs(i_es_cost_pct) > 1e-6 else float("inf")
        print(f"\n{cell}  (best-ηT = {eta_t}m):")
        print(f"  iter_192 ES-cost  : {i_es_cost_abs:+.2f} GWh ({i_es_cost_pct:+.4f}%)")
        print(f"  baseline ES-cost  : {b_es_cost_abs:+.2f} GWh ({b_es_cost_pct:+.4f}%)")
        print(f"  baseline / iter_192 ES-cost ratio: {ratio:.2f}×")
        asym_rows.append({"cell": cell, "i_es_pct": i_es_cost_pct,
                          "b_es_pct": b_es_cost_pct, "ratio": ratio})

    # Aggregate
    print("\nMean ES-cost across cells (% of endpoint AEP):")
    if asym_rows:
        ic = np.mean([r["i_es_pct"] for r in asym_rows])
        bc = np.mean([r["b_es_pct"] for r in asym_rows])
        print(f"  iter_192: {ic:+.4f}%   |   baseline: {bc:+.4f}%")
        if abs(bc) >= 3 * abs(ic):
            print(f"  → ASYMMETRY: baseline ES-cost is {abs(bc/ic):.1f}× iter_192's")
        elif abs(bc - ic) < 0.05:
            print(f"  → NO ASYMMETRY: ES-costs comparable")
        else:
            print(f"  → MILD ASYMMETRY: baseline {abs(bc/ic):.1f}× iter_192's")

    # Task C two-curve figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (cell, eta_t) in zip(axes, CELLS):
        i_off = per_iter_means(load(cell, "iter192_off"))
        i_on = per_iter_means(load(cell, "iter192_on"))
        b_off = per_iter_means(load(cell, f"baseline_eta{eta_t}_off"))
        b_on = per_iter_means(load(cell, f"baseline_eta{eta_t}_on"))
        if not all([i_off, i_on, b_off, b_on]):
            continue
        ax.plot(i_off["iters"], i_off["aep"], color="#c73a1a", linewidth=1.6, label="iter_192 ES-off")
        ax.plot(i_on["iters"], i_on["aep"], color="#c73a1a", linewidth=1.6, linestyle="--", label="iter_192 ES-on")
        ax.plot(b_off["iters"], b_off["aep"], color="#1a73c7", linewidth=1.6, label=f"baseline ES-off (ηT={eta_t}m)")
        ax.plot(b_on["iters"], b_on["aep"], color="#1a73c7", linewidth=1.6, linestyle="--", label="baseline ES-on")
        ax.axvline(ITER_192_REAL_ES_TRIGGER, color="green", linestyle=":", linewidth=0.8)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("AEP (GWh)")
        ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(TRACE_DIR, "fig_mechanism_4curve.pdf"))
    plt.savefig(os.path.join(TRACE_DIR, "fig_mechanism_4curve.png"), dpi=180)
    plt.close()
    print(f"\nWrote fig_mechanism_4curve.{{pdf,png}}")

    # Ratio-vs-iter figure (the discriminator visualisation)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (cell, eta_t) in zip(axes, CELLS):
        i_off = per_iter_means(load(cell, "iter192_off"))
        b_off = per_iter_means(load(cell, f"baseline_eta{eta_t}_off"))
        if i_off is None or b_off is None:
            continue
        ax.semilogy(i_off["iters"], i_off["ratio_obj_over_alpha_con"], color="#c73a1a", linewidth=1.6, label="iter_192")
        ax.semilogy(b_off["iters"], b_off["ratio_obj_over_alpha_con"], color="#1a73c7", linewidth=1.6, label=f"baseline (ηT={eta_t}m)")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="r = 1 (gradients balanced)")
        ax.axvline(ITER_192_REAL_ES_TRIGGER, color="green", linestyle=":", linewidth=0.8, label="ES fires (iter ≈ 6400)")
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"r = $\|\nabla\!_{xy}\,AEP\|\,/\,(\alpha\,\|\nabla\!_{xy}\,\gamma\|)$ (log)")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(TRACE_DIR, "fig_gradient_balance.pdf"))
    plt.savefig(os.path.join(TRACE_DIR, "fig_gradient_balance.png"), dpi=180)
    plt.close()
    print(f"Wrote fig_gradient_balance.{{pdf,png}}")


if __name__ == "__main__":
    main()
