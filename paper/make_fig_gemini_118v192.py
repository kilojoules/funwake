"""Compare two Gemini schedules: attempt 118 (best held-out validation) vs
attempt 192 (the file labelled 'deployed'). Plots lr, alpha, beta1, beta2 over
the 8000-step SGD run for each. Nominal lr0=50, alpha0=1 (shape comparison).

Output: paper/figs/fig_gemini_118v192.{pdf,png}
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUT = "#333333", "#777777"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "sans-serif",
})


def load_fn(path):
    spec = importlib.util.spec_from_file_location("s", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.schedule_fn


TOTAL = 8000
LR0, A0 = 50.0, 1.0
steps = np.arange(TOTAL)

SCHEDS = [
    ("iter 118  (best validation, 4269 GWh)", "#2c6fbb",
     "results_agent_gemini_cli_5hr/iter_118.py"),
    ("iter 192  (labelled 'deployed')", "#c0392b",
     "results_agent_gemini_cli_5hr/iter_192.py"),
]

data = {}
for label, color, rel in SCHEDS:
    fn = load_fn(os.path.join(ROOT, rel))
    out = np.array([np.asarray(fn(int(s), TOTAL, LR0, A0), float) for s in steps])
    data[label] = (color, out)   # columns: lr, alpha, beta1, beta2

PARAMS = ["learning rate $\\eta$", "penalty $\\alpha$",
          r"$\beta_1$ (momentum)", r"$\beta_2$"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
for j, (ax, pname) in enumerate(zip(axes.ravel(), PARAMS)):
    for label, (color, out) in data.items():
        ax.plot(steps, out[:, j], color=color, lw=1.6, label=label)
    ax.set_title(pname, fontsize=9, weight="bold", color=INK, pad=3)
    ax.set_xlabel("SGD step", fontsize=7.8)
    ax.grid(alpha=0.18, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=7)
    ax.set_xlim(0, TOTAL)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07),
           ncol=2, fontsize=8, frameon=False)
fig.suptitle("Gemini CLI — schedule 118 vs 192", fontsize=10, weight="bold",
             y=1.14)

for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "paper/figs", f"fig_gemini_118v192.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
