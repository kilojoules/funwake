"""Schedule: damped farm-adaptive penalty/relax/spike sizing on top of native decay.

Changes vs. the parent (packing-density-adaptive relax/spike/hold, unbounded
sensitivity to the size proxy):

  1. all farm-invariant machinery is unchanged: self-contained inverse-time
     product lr decay to gamma_min (log-gamma bisection, cached table), short
     linear warmup inside the constant-lr plateau, and the beta1/beta2
     phase-transition tied to progress;

  2. the packing-density size proxy is unchanged in definition
         density = n_turbines / (min_spacing / D)^2
     (turbines per unit of spacing-normalized area, log-compressed and
     clipped to [-1, 1] against a mid-scale reference).

  3. NEW -- the parent's per-farm scores show BOTH extremes of the size
     proxy underperforming: the very sparse farm (parque_n10_omnidir, size
     strongly negative -> lots of early relax, weak terminal spike) AND the
     very dense farm (dei_n80_omnidir, size strongly positive -> little
     early relax, strong terminal spike) score below native, while a
     MODERATELY sparse farm (parque_n20, size mildly negative) scores far
     above native. This is the signature of an adaptation that is too
     aggressive at the extremes rather than wrong in direction: dial back
     its magnitude with a fixed damping factor _SIZE_DAMP < 1 applied to
     the size proxy before it feeds hold_frac / relax0 / spike0. This pulls
     extreme farms (n=10, n=80) back toward the safer, better-conditioned
     native-like defaults while still letting mid-range farms (n=20, n=50)
     benefit from adaptation, just proportionally less. Damping is applied
     uniformly (not farm-specific), so it generalizes to any n_turbines/D/
     min_spacing combination rather than hardcoding a particular farm.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_BASE, _RELAX_SPAN = 0.45, 0.15   # relax0 = base + span * size in [-1,1]
_RELAX_P0, _RELAX_P1 = 0.08, 0.45
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.8     # spike0 = base + span * size
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.05   # exploration plateau length
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)
_SIZE_DAMP = 0.6                # shrink adaptation magnitude at size extremes

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
    """Log-compressed, bounded, damped packing-density factor in [-1, 1].

    density = n_turbines / (min_spacing/D)^2 estimates turbines per unit of
    spacing-normalized area; larger => more spacing conflicts to manage.
    The raw log-ratio is damped toward zero so extreme farms (very sparse or
    very dense relative to the reference) get a smaller adaptation than a
    naive linear mapping would give them.
    """
    spacing_ratio = max(float(min_spacing) / max(float(D), 1e-9), 1e-6)
    density = max(float(n_turbines), 1e-6) / (spacing_ratio ** 2)
    size = math.log(density / _DENSITY_REF) / math.log(4.0)
    size = max(-1.0, min(1.0, size))
    return _SIZE_DAMP * size


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

    relax0 = _RELAX_BASE + _RELAX_SPAN * size        # sparse farms -> freer early
    spike0 = _SPIKE_BASE + _SPIKE_SPAN * size        # dense farms -> stronger late fix
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, _RELAX_P1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2