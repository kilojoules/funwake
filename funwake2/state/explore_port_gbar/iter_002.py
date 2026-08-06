"""Schedule: warm-started native decay + graduated / terminally-spiked penalty.

Changes vs. the parent (native TopFarm-SGD coupling):
  1. self-contained inverse-time product decay (log-gamma bisection, no pixwake
     import) with a cached lookup table, so lr hits gamma_min exactly on the
     final step instead of one step late;
  2. short linear warmup inside the constant-lr exploration plateau;
  3. graduated penalty: alpha is *relaxed* (0.45x native) while the layout
     spreads, restored to native by mid-run, then given a terminal x3
     feasibility-restoration spike over the last 10% of steps.  Because
     beta2 is small the update is sign-like, so alpha only re-orients the
     step (objective vs. constraint) and never inflates its length --
     spiking it late is cheap insurance for feasibility;
  4. Adam moments phase-transition with the penalty phase: (0.1, 0.2) during
     exploration -> (0.25, 0.45) during the feasibility/refinement phase, to
     average out the stochastic wind-direction sampling noise.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_HOLD_FRAC = 1.0 / 3.0          # constant-lr exploration plateau
_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX = 0.45                   # early relaxation (graduated constraints)
_RELAX_P0, _RELAX_P1 = 0.08, 0.45
_SPIKE = 2.0                    # terminal restoration -> (1 + _SPIKE) x native
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

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


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)

    lr0 = _C * diam                                  # exploration lr from D
    n_hold = max(int(n_total * _HOLD_FRAC), 1)
    n_decay = max(n_total - n_hold, 2)
    n_warm = max(int(n_total * _WARM_FRAC), 1)

    table = _decay_table(lr0, g_min, n_decay)

    s = jnp.asarray(step)
    k = jnp.clip(s - n_hold, 0, n_decay - 1)
    lr_base = jnp.take(table, k)                     # lr0 -> gamma_min

    warm = _WARM_LO + (1.0 - _WARM_LO) * jnp.clip(s / n_warm, 0.0, 1.0)
    lr = lr_base * warm

    p = s * (1.0 / n_total)                          # progress in [0, 1)
    relax = _RELAX + (1.0 - _RELAX) * _ramp(p, _RELAX_P0, _RELAX_P1)
    spike = 1.0 + _SPIKE * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2