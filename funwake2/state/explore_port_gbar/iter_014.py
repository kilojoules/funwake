"""Schedule: density-adaptive *timing* (not just magnitude) of the penalty
phase transitions, on top of the farm-adaptive relax/spike/hold parent.

The parent already scales relax0/spike0/hold_frac by a packing-density
proxy, but the *ramp windows themselves* (when tightening starts, when the
terminal spike kicks in, when Adam momentum ramps up) were fixed constants
shared by every farm. That means a sparse farm (few spacing conflicts) still
starts tightening constraints at the same fractional progress as a dense
farm (many conflicts), even though a sparse farm has more room to explore
freely and a dense farm needs earlier feasibility engagement to avoid
tangles the terminal spike alone can't fix.

Changes vs. the parent (magnitude-only density adaptation):

  1. `relax` ramp window (_RELAX_P0/_RELAX_P1) now also shifts with density:
     denser farms start tightening EARLIER (smaller p0/p1), sparser farms
     get a LONGER free-exploration window before tightening begins
     (larger p0/p1) -- "delay the ramp" for farms that can afford it, per
     the graduated/filter-method literature, instead of a single fixed
     onset for every farm.

  2. `spike` ramp window (_SPIKE_P0) shifts the same way but with a smaller
     span, since the terminal feasibility-restoration spike is inherently
     short and shouldn't move as much as the main tightening window.

  3. The beta1/beta2 Adam-moment ramp is no longer an independently tuned
     window; it now spans exactly the "damping window" between the end of
     the relax ramp and the start of the spike ramp (phase-transition the
     Adam moments with the alpha phase, not on a separate clock), so
     momentum builds precisely while the penalty is actively engaging and
     settles before the terminal spike, for every farm shape.

  All farm-invariant machinery (self-contained inverse-time product lr
  decay to gamma_min via cached log-gamma bisection table, short linear
  warmup, magnitude-only relax0/spike0/hold_frac density scaling) is
  unchanged from the parent.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty MAGNITUDE shaping, multiplying the native alpha = mean|grad J| / lr
_RELAX_BASE, _RELAX_SPAN = 0.45, 0.15   # relax0 = base + span * size in [-1,1]
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.8     # spike0 = base + span * size

# penalty TIMING shaping (new): ramp windows shift with density too.
# size > 0 (dense) -> earlier onset; size < 0 (sparse) -> later onset,
# i.e. sparse farms get a longer free-exploration window before tightening.
_RELAX_P0_BASE, _RELAX_P0_SPAN = 0.08, 0.05   # p0 in ~[0.03, 0.13]
_RELAX_P1_BASE, _RELAX_P1_SPAN = 0.45, 0.08   # p1 in ~[0.37, 0.53]
_SPIKE_P0_BASE, _SPIKE_P0_SPAN = 0.90, 0.04   # p0 in ~[0.86, 0.94]
_SPIKE_P1 = 1.00

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.05   # exploration plateau length
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)

# Adam moments: native early -> mildly averaged late, ramped across the
# "damping window" between the end of the relax ramp and the spike onset
# (computed per-farm below, not fixed constants).
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

    # density-adaptive ramp windows (Python floats, computed at trace time)
    relax_p0 = min(max(_RELAX_P0_BASE - _RELAX_P0_SPAN * size, 0.02), 0.20)
    relax_p1 = min(max(_RELAX_P1_BASE - _RELAX_P1_SPAN * size, relax_p0 + 0.10), 0.60)
    spike_p0 = min(max(_SPIKE_P0_BASE - _SPIKE_P0_SPAN * size, relax_p1 + 0.10), _SPIKE_P1 - 0.02)

    relax0 = _RELAX_BASE + _RELAX_SPAN * size        # sparse farms -> freer early
    spike0 = _SPIKE_BASE + _SPIKE_SPAN * size        # dense farms -> stronger late fix
    relax = relax0 + (1.0 - relax0) * _ramp(p, relax_p0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, spike_p0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    # Adam moments ramp exactly across the damping window between the end
    # of constraint tightening and the start of the terminal spike, so
    # momentum builds while the penalty engages and settles before the
    # spike, for every farm shape (not a separately tuned fixed window).
    r = _ramp(p, relax_p1, spike_p0)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2