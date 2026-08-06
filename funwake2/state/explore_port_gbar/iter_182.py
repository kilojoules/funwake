"""Schedule: builds on the best-so-far variant (+0.1959% farm-balanced).

Diagnosis: dei_n50 and dei_n50_uniform have IDENTICAL n_turbines/D/min_spacing,
so `_packing_size`/`_n_scale` give them the exact same schedule -- yet
dei_n50_uniform scores far worse (+0.0139%) than dei_n50 (+0.0277%), and both
sit well below dei_n80_omnidir's (+0.0257%) N~=60-100+ "high_n" relief band
(scale_pos: n=50 -> 0.368, n=80 -> 0.708, gate was P0=0.5-0.78). That gate was
already shown to help large-many-pair farms (longer hold, delayed constraint
engagement, softened terminal spike, higher alpha floor); dei_n50/uniform
never receive any of it because 0.368 < 0.5.

This variant only lowers the gate's onset (_ALPHA_FLOOR_HIGH_N_P0: 0.5 -> 0.2)
so dei_n50/dei_n50_uniform now fall partway into the ramp (~20% of full
relief at scale_pos=0.368, up from 0%) while dei_n80_omnidir gets slightly
more (ramp rises from ~0.84 to ~0.93 at scale_pos=0.708, same direction as
before). parque_n10/parque_n20 are completely unaffected: `high_n_ramp` uses
`max(_n_scale(n), 0.0)`, and both farms have n < N_REF=30, so their scale_pos
clamps to exactly 0 regardless of where P0 sits -- the ramp stays 0. Since
parque_n20 alone supplies the majority of the current farm-balanced mean,
this keeps that contribution untouched while giving the two weakest farms a
mechanism already validated on a related farm. Nothing else changes.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_BASE, _RELAX_SPAN = 0.45, 0.11   # relax0 = base + span * size in [-1,1]
_RELAX_SCALE_SPAN = -0.05       # lower early-exploration penalty floor for large-N farms
_RELAX_P0 = 0.08
_RELAX_P1_BASE, _RELAX_P1_SPAN = 0.45, 0.10   # widen engagement window at extremes
_RELAX_P1_SCALE_SPAN = 0.08     # further delay for large-N farms
_RELAX_P1_HIGH_N_SPAN = 0.05    # extra gated delay for N~=60-100+ (omnidir-heavy) farms
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.55    # spike0 = base + span * size
_SPIKE_SCALE_SPAN = -0.4        # soften terminal spike for large-N farms
_SPIKE_HIGH_N_SPAN = -0.15      # extra gated softening for N~=60-100+ band
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00
_SPIKE_P0_HIGH_N_SPAN = 0.03    # delay spike onset a bit further for that band

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.06   # exploration plateau length
_HOLD_FRAC_SCALE_SPAN = 0.05    # extra hold for large-N farms
_HOLD_FRAC_HIGH_N_SPAN = 0.02   # extra gated hold for N~=60-100+ farms
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)
_N_REF = 30.0                   # absolute-N reference (independent of density)

# Bound the terminal alpha denominator (lr_base floor) so alpha's 1/lr_base
# divergence is capped instead of fully re-engaging native's
# constraint-dominated tail on crowded (many-pair) or unusually-spaced
# layouts.
_ALPHA_FLOOR_SCALE_SPAN = 2.0
_ALPHA_FLOOR_EXT_SPAN = 0.3
# Gated relief for farms in the N ~= 20-100+ band (scale_pos in
# [0.2, 0.78]) -- widened from [0.5, 0.78] so dei_n50/dei_n50_uniform
# (scale_pos~=0.368) receive partial relief too, not just dei_n80_omnidir
# (scale_pos~=0.708). parque_n10/n20 stay at scale_pos=0 (clamped) so this
# gate never touches them.
_ALPHA_FLOOR_HIGH_N_P0 = 0.2
_ALPHA_FLOOR_HIGH_N_P1 = 0.78
_ALPHA_FLOOR_HIGH_N_SPAN = 3.5

# Adam moments: native early -> mildly averaged late
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.45
_BETA_P0, _BETA_P1 = 0.50, 0.90
_BETA_P0_SCALE_SPAN = -0.10     # earlier averaging onset for large-N farms

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


def _smoothstep_scalar(x, p0, p1):
    """Python-float (trace-time) smoothstep from 0 at p0 to 1 at p1."""
    t = max(0.0, min(1.0, (x - p0) / max(p1 - p0, 1e-12)))
    return t * t * (3.0 - 2.0 * t)


def _packing_size(n_turbines, D, min_spacing):
    """Log-compressed, bounded packing-density factor in [-1, 1].

    density = n_turbines / (min_spacing/D)^2 estimates turbines per unit of
    spacing-normalized area; larger => more spacing conflicts to manage.
    """
    spacing_ratio = max(float(min_spacing) / max(float(D), 1e-9), 1e-6)
    density = max(float(n_turbines), 1e-6) / (spacing_ratio ** 2)
    size = math.log(density / _DENSITY_REF) / math.log(4.0)
    return max(-1.0, min(1.0, size))


def _n_scale(n_turbines):
    """Log-compressed, bounded ABSOLUTE-N factor in [-1, 1], independent of
    packing density -- catches large-N farms whose spacing_ratio happens to
    sit near the density reference (so `_packing_size` alone misses them)."""
    scale = math.log(max(float(n_turbines), 1e-6) / _N_REF) / math.log(4.0)
    return max(-1.0, min(1.0, scale))


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)

    size = _packing_size(n_turbines, diam, min_spacing)
    extremity = abs(size)                              # 0 at mid-density, ->1 at extremes
    scale_pos = max(_n_scale(n_turbines), 0.0)          # 0 unless large absolute N
    high_n_ramp = _smoothstep_scalar(
        scale_pos, _ALPHA_FLOOR_HIGH_N_P0, _ALPHA_FLOOR_HIGH_N_P1)

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = (_HOLD_FRAC_BASE + _HOLD_FRAC_SPAN * extremity
                 + _HOLD_FRAC_SCALE_SPAN * scale_pos     # longer plateau at extremes AND large N
                 + _HOLD_FRAC_HIGH_N_SPAN * high_n_ramp)  # extra gated hold for N~=20-100+ band
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

    relax0 = (_RELAX_BASE + _RELAX_SPAN * size
              + _RELAX_SCALE_SPAN * scale_pos)        # lower early floor for large-N farms
    spike0 = (_SPIKE_BASE + _SPIKE_SPAN * size
              + _SPIKE_SCALE_SPAN * scale_pos
              + _SPIKE_HIGH_N_SPAN * high_n_ramp)     # extra gated softening for high-N band
    relax_p1 = (_RELAX_P1_BASE + _RELAX_P1_SPAN * extremity
                + _RELAX_P1_SCALE_SPAN * scale_pos      # further delay for large-N farms
                + _RELAX_P1_HIGH_N_SPAN * high_n_ramp)  # extra gated delay for N~=20-100+ band
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike_p0 = _SPIKE_P0 + _SPIKE_P0_HIGH_N_SPAN * high_n_ramp  # delay spike onset for high-N
    spike = 1.0 + spike0 * _ramp(p, spike_p0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha_denom_floor = g_min * (1.0 + _ALPHA_FLOOR_SCALE_SPAN * scale_pos
                                  + _ALPHA_FLOOR_EXT_SPAN * extremity
                                  + _ALPHA_FLOOR_HIGH_N_SPAN * high_n_ramp)
    alpha_denom = jnp.maximum(lr_base, alpha_denom_floor)
    alpha = alpha0 * diam * relax * spike / alpha_denom

    # Never let the beta ramp start before the lr hold phase ends -- for
    # large-N/extreme-density farms, _BETA_P0_SCALE_SPAN pulls beta_p0 down
    # while the hold_frac terms push hold_frac up, and the two can cross so
    # momentum starts increasing during pure exploration (before lr has
    # decayed at all) on exactly the crowded, many-pair farms that most need
    # unaveraged gradient steps there.
    beta_p0 = max(_BETA_P0 + _BETA_P0_SCALE_SPAN * scale_pos, hold_frac)
    r = _ramp(p, beta_p0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2