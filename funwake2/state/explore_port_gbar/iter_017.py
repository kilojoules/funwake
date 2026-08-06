"""Schedule: bounded-alpha plateau on top of the extremity-aware parent.

Change vs. the parent (packing-density-adaptive relax/spike/hold):

  The parent couples alpha to the *native* 1/lr law: alpha = alpha0 * D *
  relax * spike / lr_base. Because lr_base decays all the way down to
  gamma_min (a metre-scale tolerance that can be orders of magnitude smaller
  than lr0), this coupling is effectively unbounded in the terminal phase --
  exactly the "diverging coupling" the design survey flags as the weakest
  part of the native scheme (frontier menu item 1: bounded/logistic alpha
  plateau instead of alpha proportional to 1/lr -> infinity).

  dei_n80_omnidir (the farm that regresses below native) has more turbines
  than the density-reference farm, so both relax0 and spike0 (which grow
  with the *signed* packing-density proxy `size`) push it toward the
  largest terminal spike among all farms -- on top of an lr_base that is
  already tiny near the end. The two effects compound multiplicatively and
  the resulting alpha spike is likely far stiffer than needed, overpowering
  the AEP gradient exactly when fine positioning matters most.

  Fix: replace the raw 1/lr_base coupling with a smooth, extremity-scaled
  *soft cap* (harmonic saturation: alpha_cap*alpha_native/(alpha_cap +
  alpha_native)). This leaves alpha essentially unchanged whenever the
  native coupling is well below the cap (i.e. for most of training, and for
  farms whose native alpha never gets that large), but prevents the
  terminal blow-up as lr_base -> gamma_min. The cap itself grows mildly with
  |size| so farms that legitimately need a stronger terminal push (already
  identified via the existing extremity machinery) still get a
  proportionally larger ceiling -- just a *bounded* one instead of an
  unbounded one. This is a decoupled, better-conditioned generalization of
  the native law, not a farm-specific hack: it is expressed purely in terms
  of alpha0, D, and the existing size/extremity proxies, so it adapts to
  any N / spacing / geometry.

  All other machinery (self-contained inverse-time product lr decay to
  gamma_min, short linear warmup, extremity-aware hold/relax/spike shaping,
  beta1/beta2 phase-transition) is unchanged from the parent.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_BASE, _RELAX_SPAN = 0.45, 0.11   # relax0 = base + span * size in [-1,1]
_RELAX_P0 = 0.08
_RELAX_P1_BASE, _RELAX_P1_SPAN = 0.45, 0.10   # widen engagement window at extremes
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.55    # spike0 = base + span * size
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

# bounded alpha plateau: soft-caps the native alpha0*D*relax*spike/lr coupling
# so it saturates instead of diverging as lr_base -> gamma_min.
_ALPHA_CAP_MULT = 20.0          # cap, in units of alpha0*D, at zero extremity
_ALPHA_CAP_EXT = 1.0            # cap grows up to 2x at max extremity

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

    relax0 = _RELAX_BASE + _RELAX_SPAN * size        # sparse farms -> freer early
    spike0 = _SPIKE_BASE + _SPIKE_SPAN * size        # dense farms -> stronger late fix
    relax_p1 = _RELAX_P1_BASE + _RELAX_P1_SPAN * extremity   # extreme farms: delay full penalty
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha_native = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    # bounded plateau: soft-cap the native coupling so it saturates instead
    # of diverging as lr_base -> gamma_min. Cap scales mildly with extremity
    # so farms whose relax/spike shaping already targets a larger terminal
    # push retain proportionally more headroom, without ever exploding.
    alpha_cap = _ALPHA_CAP_MULT * alpha0 * diam * (1.0 + _ALPHA_CAP_EXT * extremity)
    alpha = alpha_cap * alpha_native / jnp.maximum(alpha_cap + alpha_native, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2