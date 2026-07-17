"""Analyse per-iter AEP traces from Task 1.

Mechanism question: does ES-off AEP keep rising AFTER the ES trigger point?
- yes → late-tail interpretation supported (ES truncates productive optimization)
- no → endpoint effect has different cause

For iter_192, the precomputed `es_first_cross_iter` reports iter 0 (warmup
spurious — lr=0/runmax=1e-10 = 0 ≤ 0.1). The 'real' running-max ES trigger
fires when lr/lr_peak ≤ 0.1, i.e. lr=20 in the cosine decay phase, around
iter ≈ 6440 for iter_192. We mark BOTH on the figure but interpret around
the real trigger.

For each mechanism cell:
- read ES-off + ES-on traces, all 3 sample seeds
- AEP delta = mean(ES-off[trigger:end]) − mean(ES-off[trigger])
- If positive → ES-off still rising → late-tail interpretation supports
- Per-iter delta plot ES-on vs ES-off
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = "/Users/julianquick/portfolio_copy/funwake"
TRACE_DIR = os.path.join(REPO, "validation/stochastic_aep")

# iter_192 schedule: lr_peak = 200, decays via cosine after warmup_end=0.05.
# "Real" running-max ES crossing: lr/lr_peak ≤ 0.1 → lr ≤ 20.
# From cosine inverse: t = 0.805 → iter ≈ 6440.
ITER_192_REAL_ES_TRIGGER = 6440
ITER_192_LR_PEAK = 200.0


MECHANISM_CELLS = [
    "rowp_n80_roserowp",
    "rowp_n80_roseomnidir",
    "rowp_n70_roserowp",
]
CONVERGENCE_CELLS = ["dei_n50_rosedei", "rowp_n80_roserowp"]


def load_trace(cell, es_mode):
    p = os.path.join(TRACE_DIR, f"per_iter_{cell}_{es_mode}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def mean_trace(d):
    """Mean across sample seeds (per iter)."""
    runs = [r for r in d["runs"] if "error" not in r]
    if not runs:
        return None, None
    iter_trace = np.asarray(runs[0]["iter_trace"])
    aeps = np.stack([r["aep_trace_gwh"] for r in runs])  # (n_seeds, n_iters)
    return iter_trace, aeps


def main():
    print("=" * 80)
    print("MECHANISM (per-cell ES-off vs ES-on)")
    print("=" * 80)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, cell in zip(axes, MECHANISM_CELLS):
        off = load_trace(cell, "off")
        on = load_trace(cell, "on_runmax")
        if off is None or on is None:
            print(f"{cell}: missing trace"); continue
        it_off, off_aeps = mean_trace(off)
        it_on, on_aeps = mean_trace(on)

        off_mean = off_aeps.mean(0)
        off_std = off_aeps.std(0, ddof=1) if off_aeps.shape[0] > 1 else np.zeros_like(off_mean)
        on_mean = on_aeps.mean(0)
        on_std = on_aeps.std(0, ddof=1) if on_aeps.shape[0] > 1 else np.zeros_like(on_mean)

        # Pre-reg metric: ES-off rise post-trigger
        trigger_idx = int(np.argmin(np.abs(it_off - ITER_192_REAL_ES_TRIGGER)))
        aep_at_trigger = off_mean[trigger_idx]
        aep_at_end = off_mean[-1]
        post_trigger_delta = aep_at_end - aep_at_trigger
        post_trigger_pct = 100.0 * post_trigger_delta / aep_at_trigger

        # ES-on at end vs ES-off at end
        endpoint_delta = on_mean[-1] - off_mean[-1]
        endpoint_pct = 100.0 * endpoint_delta / off_mean[-1]

        print(f"\n{cell}:")
        print(f"  AEP @ trigger iter {it_off[trigger_idx]} (ES-off): {aep_at_trigger:.2f} GWh")
        print(f"  AEP @ end (ES-off):                                  {aep_at_end:.2f} GWh")
        print(f"  Post-trigger ES-off rise:                            {post_trigger_delta:+.2f} GWh ({post_trigger_pct:+.3f}%)")
        print(f"  Endpoint ES-on − ES-off:                             {endpoint_delta:+.2f} GWh ({endpoint_pct:+.3f}%)")
        if post_trigger_pct > 0.05 and endpoint_pct < -0.05:
            print(f"  → Pre-reg rule: late-tail interpretation SUPPORTED (rise > 0.05, endpoint < -0.05)")
        elif post_trigger_pct > 0.05:
            print(f"  → ES-off rising post-trigger but endpoint not as negative — partial support")
        elif post_trigger_pct <= 0.05:
            print(f"  → Pre-reg rule: late-tail interpretation NOT supported (no significant rise)")

        ax.plot(it_off, off_mean, color="#1a73c7", linewidth=1.6, label="ES-off")
        ax.fill_between(it_off, off_mean - off_std, off_mean + off_std,
                          color="#1a73c7", alpha=0.2)
        ax.plot(it_on, on_mean, color="#c73a1a", linewidth=1.6, label="ES-on (running-max)")
        ax.fill_between(it_on, on_mean - on_std, on_mean + on_std,
                          color="#c73a1a", alpha=0.2)
        ax.axvline(ITER_192_REAL_ES_TRIGGER, color="green", linestyle="--", linewidth=0.8,
                    label=f"real ES trigger (iter≈{ITER_192_REAL_ES_TRIGGER})")
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("AEP (GWh, deterministic eval)")
        ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(TRACE_DIR, "fig_mechanism.pdf"))
    plt.savefig(os.path.join(TRACE_DIR, "fig_mechanism.png"), dpi=180)
    plt.close()
    print(f"\nWrote fig_mechanism.{{pdf,png}}")

    # Convergence figure
    print("\n" + "=" * 80)
    print("CONVERGENCE (training vs held-out, ES-off)")
    print("=" * 80)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    colors = {"dei_n50_rosedei": "#1a73c7", "rowp_n80_roserowp": "#c73a1a"}
    labels = {"dei_n50_rosedei": "DEI N=50 training", "rowp_n80_roserowp": "ROWP N=80 held-out"}
    for cell in CONVERGENCE_CELLS:
        off = load_trace(cell, "off")
        if off is None:
            print(f"  {cell}: missing"); continue
        it_off, off_aeps = mean_trace(off)
        m = off_aeps.mean(0)
        s = off_aeps.std(0, ddof=1) if off_aeps.shape[0] > 1 else np.zeros_like(m)
        # Plot as normalized to final to overlay both
        norm = m / m[-1]
        ax.plot(it_off, norm, color=colors[cell], linewidth=1.6, label=labels[cell])
        ax.fill_between(it_off, norm - s/m[-1], norm + s/m[-1],
                         color=colors[cell], alpha=0.2)
        print(f"  {cell}: AEP[0]={m[0]:.2f}, AEP[end]={m[-1]:.2f}, gain={(m[-1]-m[0])/m[0]*100:.2f}%")
    ax.axvline(ITER_192_REAL_ES_TRIGGER, color="green", linestyle="--", linewidth=0.8,
                label=f"ES trigger (iter≈{ITER_192_REAL_ES_TRIGGER})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AEP / final AEP (normalised)")
    ax.set_title("iter_192 ES-off convergence (Claude-evolved schedule)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(TRACE_DIR, "fig_convergence.pdf"))
    plt.savefig(os.path.join(TRACE_DIR, "fig_convergence.png"), dpi=180)
    plt.close()
    print(f"\nWrote fig_convergence.{{pdf,png}}")


if __name__ == "__main__":
    main()
