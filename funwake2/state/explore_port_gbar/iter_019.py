"""Schedule: asymmetric penalty shaping -- soften dense-farm constraint pressure.

Changes vs. the parent (packing-density-adaptive relax/spike/hold via a single
symmetric |size| extremity term):

  The remaining regression is dei_n80_omnidir (dense/large-N, size > 0), while
  the sparse/small-N extreme (parque_n10_omnidir, size < 0) was already fixed
  by the parent's symmetric hold_frac/relax-window widening. dei_n80_omnidir
  is already feasible (3/3 seeds) under native, so the parent's *dense-side*
  shaping -- which INCREASES both the early alpha floor (relax0 grows with
  +size) and the terminal alpha spike (spike0 grows with +size) -- is adding
  constraint pressure the farm doesn't need to reach feasibility, and that
  pressure is what's costing it AEP relative to native.

  Fix: split _RELAX_SPAN and _SPIKE_SPAN into separate positive-size (dense)
  and negative-size (sparse) sensitivities. The sparse-side values are left
  exactly as in the parent (that combination already improved
  parque_n10_omnidir to +0.22%). The dense-side sensitivities are cut
  substantially (~2x on relax, ~2x on spike), so dense/large-N farms still
  get the longer exploration plateau (hold_frac, unchanged) but the
  constraint machinery ramps and spikes much more gently once alpha engages,
  leaving more of the extended exploration gains intact instead of being
  clawed back by an aggressive terminal correction.

  size/extremity are computed from static (non-traced) D/min_spacing/
  n_turbines, so branching on their sign at trace time is safe -- no Python
  branch depends on `step` or `alpha0`.

  All other machinery (self-contained inverse-time product lr decay to
  gamma_min, short linear warmup, symmetric extremity-widened hold/relax
  window, beta1/beta2 phase-transition) is unchanged from the parent.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_BASE = 0.45              # relax0 = base + span * size in [-1,1]
_RELAX_SPAN_NEG = 0.11          # sparse (size<0): unchanged from parent
_RELAX_SPAN_POS = 0.04          # dense (size>0): softened -- less early alpha
_RELAX_P0 = 0.08
_RELAX_P1_BASE, _RELAX_P1_SPAN = 0.45, 0.10   # widen engagement window at extremes
_SPIKE_BASE = 2.0               # spike0 = base + span * size
_SPIKE_SPAN_NEG = 0.55          # sparse (size<0): unchanged from parent
_SPIKE_SPAN_POS = 0.28          # dense (size>0): softened -- gentler terminal fix
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.06   # exploration plateau length
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)

# Adam moments: native early -> mildly averaged late
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.45
_BETA_P0, _BETA_P1 = 0.50, 0.90

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
    extremity = abs(size)                              # 0 at mid-density, ->1 at extremes

    # dense (size>0) vs sparse (size<0) get separate shaping sensitivities;
    # this branches on a static value (size), never on step/alpha0.
    if size >= 0.0:
        relax_span = _RELAX_SPAN_POS
        spike_span = _SPIKE_SPAN_POS
    else:
        relax_span = _RELAX_SPAN_NEG
        spike_span = _SPIKE_SPAN_NEG

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = _HOLD_FRAC_BASE + _HOLD_FRAC_SPAN * extremity   # longer plateau at BOTH extremes
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

    relax0 = _RELAX_BASE + relax_span * size         # sparse farms -> freer early
    spike0 = _SPIKE_BASE + spike_span * size         # dense farms -> stronger (but bounded) late fix
    relax_p1 = _RELAX_P1_BASE + _RELAX_P1_SPAN * extremity   # extreme farms: delay full penalty
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2