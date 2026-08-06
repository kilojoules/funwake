"""Schedule: self-contained (no pixwake import) product-decay lr with a short
warmup, plus a decoupled, bounded alpha shaping and a delayed beta ramp that
both adapt to farm size/density via n_turbines, D, and min_spacing.

Diagnosis vs. the gen-108 parent: the parent recovers native's alpha = mean|
grad J| / lr coupling exactly, which means alpha diverges to infinity as lr
decays toward gamma_min on every farm alike -- there is no per-farm lever, so
larger/denser farms (more spacing-constraint pairs) end up over-penalized
late (killing achievable AEP) while smaller farms are comparatively
under-penalized early (risking transient infeasibility). It also has zero
warmup (first step takes a full lr0 stride from the grid init) and fixed
beta1/beta2=0.1/0.2 for the whole run, so momentum never adapts as the
penalty phase engages. This edit: (1) replaces the pixwake bisection import
with a local, cached closed-form bisection so the schedule is fully
self-contained; (2) adds a short linear warmup off a low multiplier so the
first steps don't overshoot; (3) floors alpha's denominator at a
size-adaptive multiple of gamma_min (using a log-compressed, bounded
n_turbines/spacing "size" factor) so the penalty plateaus late instead of
diverging, with larger/denser farms getting a higher floor (more relief) and
smaller/sparser farms staying close to native; (4) ramps beta1/beta2 up from
the native (0.1, 0.2) toward a mildly-averaged plateau only after the lr
hold phase ends, with the ramp onset itself pulled earlier for large-N
farms. All shaping is a function of D/min_spacing/n_turbines/gamma_min only
(all Python floats at trace time), so it generalizes across farms without
hardcoding any single farm's behavior.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_HOLD_FRAC = 1.0 / 3.0           # plateau length before decay begins
_WARM_FRAC = 0.01                # linear warmup inside the plateau
_WARM_LO = 0.3                   # lr multiplier at step 0

_DENSITY_REF = 12.5              # ~50 turbines at 2D spacing (mid-scale anchor)
_ALPHA_FLOOR_SPAN = 1.5          # extra alpha-denominator headroom at max size
_BETA_P0_BASE, _BETA_P0_SPAN = 0.55, -0.15   # earlier ramp onset for large farms
_BETA_P1 = 0.92
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
    """lr0 * prod_{t<=j} 1/(1 + mid*t) for j = 0..n_decay-1, ending at gamma_min."""
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


def _size_factor(n_turbines, D, min_spacing):
    """Log-compressed, bounded packing-density factor in [0, 1].

    density = n_turbines / (min_spacing/D)^2 estimates turbines per unit of
    spacing-normalized area; larger => more spacing conflicts to manage.
    """
    spacing_ratio = max(float(min_spacing) / max(float(D), 1e-9), 1e-6)
    density = max(float(n_turbines), 1e-6) / (spacing_ratio ** 2)
    size = math.log(density / _DENSITY_REF) / math.log(4.0)
    return max(0.0, min(1.0, 0.5 * (size + 1.0)))


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)

    size = _size_factor(n_turbines, diam, min_spacing)   # 0 (sparse/small) .. 1 (dense/large)

    lr0 = _C * diam
    n_hold = max(int(n_total * _HOLD_FRAC), 1)
    n_decay = max(n_total - n_hold, 2)
    n_warm = max(int(n_total * _WARM_FRAC), 1)

    table = _decay_table(lr0, g_min, n_decay)

    s = jnp.asarray(step)
    k = jnp.clip(s - n_hold, 0, n_decay - 1)
    lr_base = jnp.take(table, k)                     # lr0 -> gamma_min

    warm = _WARM_LO + (1.0 - _WARM_LO) * jnp.clip(s / n_warm, 0.0, 1.0)
    lr = lr_base * warm

    # Bound alpha's denominator at a size-adaptive multiple of gamma_min so
    # the penalty plateaus late instead of diverging as lr_base -> gamma_min;
    # larger/denser farms (more constraint pairs) get more headroom.
    alpha_floor = g_min * (1.0 + _ALPHA_FLOOR_SPAN * size)
    alpha_denom = jnp.maximum(lr_base, alpha_floor)
    alpha = alpha0 * diam / alpha_denom

    p = s * (1.0 / n_total)                          # progress in [0, 1)
    beta_p0 = max(_BETA_P0_BASE + _BETA_P0_SPAN * size, _HOLD_FRAC)
    x = jnp.clip((p - beta_p0) / max(_BETA_P1 - beta_p0, 1e-12), 0.0, 1.0)
    r = x * x * (3.0 - 2.0 * x)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2