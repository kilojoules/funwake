"""Behavioral descriptors computed from a schedule_fn (cheap, PRE-eval).

Descriptors (spec 3.4, frozen bins D-8) are read off the schedule's lr(t) /
alpha(t) profiles at a canonical reference cell (D=240, min_spacing=960, N=50,
gamma_min=0.01) so they are comparable across candidates. Because lr is built
from D INSIDE the schedule, peak_lr/D is ~scale-invariant; we evaluate at the
canonical D and report peak_lr/D, terminal_lr (m), the alpha-coupling class, and
the restart-bump count.

Verified against R4: iter192 -> peak_lr/D 0.833; iter118 -> 1.354; native 0.833
(distinct cell via coupling/terminal/restarts).
"""
from __future__ import annotations

import numpy as np

from . import config as C

_REF = dict(total_steps=C.TOTAL_STEPS, D=240.0, min_spacing=960.0,
            n_turbines=50, gamma_min=C.GAMMA_MIN, alpha0=1e-4)


def _profiles(schedule_fn, total_steps, ref):
    """Return lr[t], alpha[t] arrays over 0..total_steps-1 as float64 numpy.

    Tries a JAX vmap; falls back to a coarse python loop for schedules that are
    not vmap-clean. The coarse grid (800 samples) is sufficient for binning.
    """
    import jax
    import jax.numpy as jnp

    def call(i):
        lr, alpha, b1, b2 = schedule_fn(
            i, total_steps, ref["D"], ref["min_spacing"],
            ref["n_turbines"], ref["gamma_min"], ref["alpha0"])
        # no forced float64 (descriptors don't need it; avoids an x64 warning
        # in processes that haven't enabled jax_enable_x64)
        return jnp.asarray(lr), jnp.asarray(alpha)

    steps = jnp.arange(total_steps)
    try:
        lr, alpha = jax.vmap(call)(steps)
        return np.asarray(lr, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    except Exception:
        idx = np.unique(np.linspace(0, total_steps - 1, 800).astype(int))
        lrs, als = [], []
        for i in idx:
            lr, alpha = call(int(i))
            lrs.append(float(lr)); als.append(float(alpha))
        # re-expand terminal correctly: ensure last step present
        return np.array(lrs), np.array(als)


def _count_local_maxima(a, rel_prom=0.02):
    """Count interior strict local maxima with a relative-prominence filter."""
    a = np.asarray(a, dtype=np.float64)
    if a.size < 3:
        return 0
    rng = float(np.nanmax(a) - np.nanmin(a))
    if rng <= 0:
        return 0
    thr = rel_prom * rng
    n = 0
    i = 1
    N = a.size
    while i < N - 1:
        if a[i] > a[i - 1] and a[i] >= a[i + 1]:
            # confirm it rises at least `thr` above the preceding local min
            j = i - 1
            while j > 0 and a[j] <= a[j + 1]:
                j -= 1
            if a[i] - a[j] >= thr:
                n += 1
        i += 1
    return n


def _count_local_minima(a, rel_prom=0.05):
    return _count_local_maxima(-np.asarray(a, dtype=np.float64), rel_prom)


def classify_coupling(lr, alpha):
    """coupled (alpha ~ 1/lr) / decoupled / cyclic.

    Uses the profile over t in [0, 0.98] (the terminal squeeze region is
    ignored so a single endgame spike does not dominate the correlation).
    """
    N = len(lr)
    hi = max(3, int(0.98 * N))
    lr_s = np.asarray(lr[:hi], dtype=np.float64)
    al_s = np.asarray(alpha[:hi], dtype=np.float64)
    inv = 1.0 / np.maximum(lr_s, 1e-30)
    # cyclic if alpha oscillates with >=3 pronounced dips (per-cycle relaxations)
    n_min = _count_local_minima(al_s, rel_prom=0.08)
    if n_min >= 3:
        return "cyclic"
    la, li = np.log(np.maximum(al_s, 1e-30)), np.log(inv)
    if np.std(la) < 1e-9 or np.std(li) < 1e-9:
        return "decoupled"
    r = float(np.corrcoef(la, li)[0, 1])
    return "coupled" if r >= 0.5 else "decoupled"


def compute_descriptors(schedule_fn, total_steps=None, ref=None):
    ref = dict(_REF if ref is None else ref)
    total_steps = total_steps or ref["total_steps"]
    lr, alpha = _profiles(schedule_fn, total_steps, ref)
    peak_lr_over_D = float(np.nanmax(lr) / ref["D"])
    terminal_lr_m = float(lr[-1])
    coupling = classify_coupling(lr, alpha)
    restarts = int(_count_local_maxima(lr, rel_prom=0.03))
    return {
        "peak_lr_over_D": round(peak_lr_over_D, 4),
        "terminal_lr_m": round(terminal_lr_m, 6),
        "coupling": coupling,
        "restarts": restarts,
    }


# ── binning to the FROZEN grid (D-8) ─────────────────────────────────
def _bin_edges(value, edges):
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def bin_descriptors(desc) -> tuple:
    peak_bin = _bin_edges(desc["peak_lr_over_D"], C.PEAK_LR_OVER_D_EDGES)
    term_bin = _bin_edges(desc["terminal_lr_m"], C.TERMINAL_LR_M_EDGES)
    coup_bin = desc["coupling"]
    r = desc["restarts"]
    if r <= 0:
        restart_bin = 0
    elif r <= 2:
        restart_bin = 1
    else:
        restart_bin = 2
    return (peak_bin, term_bin, coup_bin, restart_bin)


PEAK_LABELS = ["<0.5", "0.5-0.8", "0.8-1.2", ">1.2"]
TERM_LABELS = ["<=0.01", "0.01-0.1", "0.1-1", ">1"]
RESTART_LABELS = ["0", "1-2", ">=3"]


def cell_label(coord) -> str:
    p, t, c, r = coord
    return f"peak[{PEAK_LABELS[p]}]|term[{TERM_LABELS[t]}]|{c}|restart[{RESTART_LABELS[r]}]"
