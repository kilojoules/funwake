"""Schedule: bounded-plateau, delayed-ramp alpha with terminal lr-coupled
feasibility spike, on top of the proven farm-adaptive lr decay/warmup.

Changes vs. the parent (alpha continuously coupled to 1/lr_base, growing
monotonically for the entire decay phase because lr_base shrinks the whole
run):

  1. lr schedule, warmup, packing-density sizing, and the closed-form
     inverse-time decay table are UNCHANGED (proven: 5/5 farms feasible).

  2. alpha is now split into three regimes instead of one diverging 1/lr
     coupling:
       - early/mid (delayed ramp, p in [_MID_P0, _MID_P1]): alpha ramps
         from a small, farm-adaptive fraction of mean|grad J| up to a
         BOUNDED plateau -- also a fraction of mean|grad J|, but NOT
         divided by the shrinking lr_base, so it stays flat once the
         ramp completes instead of growing for the rest of training.
         This gives the optimizer a long, stable, moderately-constrained
         exploration phase (graduated penalty, LANCELOT/filter-style
         delayed ramp) instead of ever-tightening pressure.
       - terminal (p in [_TERM_P0, _TERM_P1]): alpha blends from the
         bounded plateau into the same lr-coupled value the parent used
         (alpha0*D*spike/lr_base), which is what actually drives the
         optimizer to gamma_min feasibility as lr_base -> gamma_min. This
         keeps the parent's proven convergence/feasibility guarantee
         while removing the mid-run divergence.
     Both plateau height and terminal spike remain keyed off the same
     packing-density "size" proxy as the parent, so sparse farms get a
     lower/later plateau (freer exploration) and dense farms get a
     higher/bigger terminal correction.

  3. beta1/beta2 phase-transition is now tied to the SAME ramp window as
     the alpha regimes (mid-ramp start -> terminal-ramp end) instead of a
     separate progress window, so Adam's averaging increases in lockstep
     with the constraint-penalty phase transition, per the ALM-style
     "increase momentum/averaging as the feasibility phase begins"
     hypothesis.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# bounded alpha plateau (fraction of mean|grad J|), farm-adaptive via `size`
_RELAX_LO_BASE, _RELAX_LO_SPAN = 0.12, 0.06     # pre-ramp (exploration) level
_RELAX_HI_BASE, _RELAX_HI_SPAN = 0.55, 0.20     # bounded plateau level
_MID_P0, _MID_P1 = 0.12, 0.50                   # delayed ramp window

# terminal blend into the lr-coupled (native) feasibility-driving value
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.8             # spike0 = base + span * size
_TERM_P0, _TERM_P1 = 0.85, 0.98                 # terminal blend window

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.05   # exploration plateau length
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)

# Adam moments: native early -> mildly averaged late, phased with alpha
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.45

_TABLE_CACHE = {}


def _log_prod(m, n):
    """sum_{t=1..n} log(1 + m*t), closed form via log-gamma (O(1))."""
    if m <= 0.0:
        return 0.0
    inv = 1.0 / m
    return n * math.log(m) + math.lgamma(inv + n + 1.0) - math.lgamma(inv + 1.0)


def _decay_table(lr0, gamma_min, n_decay):
    """lr0 * prod_{t<=j} 1/(1 + mid*t) for j = 0..n_decay-1, ending at gamma_min.

    mid is found by bisection on the (monotone) log-product; everything here is
    a Python float at trace time, so the result is a compile-time constant.
    """
    key = (lr0, gamma_min, n_decay)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    horizon = max(n_decay - 1, 1)
    target = math.log(max(lr0, 1e-30) / max(gamma_min, 1e-30))
    lo, hi = 0.0, 0.1
    while _log_prod(hi, horizon) < target and hi < 1e6:
        hi *= 4.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _log_prod(mid, horizon) < target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)

    vals, lr = [], lr0
    for t in range(n_decay):
        vals.append(lr)
        lr = lr / (1.0 + mid * (t + 1.0))
    table = jnp.asarray(vals)
    _TABLE_CACHE[key] = table
    return table


def _ramp(p, p0, p1):
    """Traceable smoothstep from 0 at p0 to 1 at p1."""
    x = jnp.clip((p - p0) / max(p1 - p0, 1e-12), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _packing_size(n_turbines, D, min_spacing):
    """Log-compressed, bounded packing-density factor in [-1, 1].

    density = n_turbines / (min_spacing/D)^2 estimates turbines per unit of
    spacing-normalized area; larger => more spacing conflicts to manage.
    """
    spacing_ratio = max(float(min_spacing) / max(float(D), 1e-9), 1e-6)
    density = max(float(n_turbines), 1e-6) / (spacing_ratio ** 2)
    size = math.log(density / _DENSITY_REF) / math.log(4.0)
    return max(-1.0, min(1.0, size))


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)

    size = _packing_size(n_turbines, diam, min_spacing)

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = _HOLD_FRAC_BASE + _HOLD_FRAC_SPAN * size
    n_hold = max(int(n_total * hold_frac), 1)
    n_decay = max(n_total - n_hold, 2)
    n_warm = max(int(n_total * _WARM_FRAC), 1)

    table = _decay_table(lr0, g_min, n_decay)

    s = jnp.asarray(step)
    k = jnp.clip(s - n_hold, 0, n_decay - 1)
    lr_base = jnp.take(table, k)                     # lr0 -> gamma_min

    warm = _WARM_LO + (1.0 - _WARM_LO) * jnp.clip(s / n_warm, 0.0, 1.0)
    lr = lr_base * warm

    p = s * (1.0 / n_total)                          # progress in [0, 1)

    # bounded, delayed-ramp alpha plateau (fraction of mean|grad J|=alpha0*D)
    relax_lo = _RELAX_LO_BASE + _RELAX_LO_SPAN * size
    relax_hi = _RELAX_HI_BASE + _RELAX_HI_SPAN * size
    mid_ramp = _ramp(p, _MID_P0, _MID_P1)
    alpha_bounded = alpha0 * diam * (relax_lo + (relax_hi - relax_lo) * mid_ramp)

    # terminal blend into the native lr-coupled value that drives feasibility
    # down to gamma_min as lr_base -> gamma_min (same mechanism as the parent).
    spike0 = _SPIKE_BASE + _SPIKE_SPAN * size
    term_ramp = _ramp(p, _TERM_P0, _TERM_P1)
    alpha_terminal = alpha0 * diam * (1.0 + spike0) / jnp.maximum(lr_base, 1e-30)

    alpha = alpha_bounded * (1.0 - term_ramp) + alpha_terminal * term_ramp

    r = _ramp(p, _MID_P0, _TERM_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2