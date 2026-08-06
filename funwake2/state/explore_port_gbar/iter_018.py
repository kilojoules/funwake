"""Schedule: asymmetric extremity handling — dense/large-N farms get a longer
exploration hold and a softer terminal spike than sparse farms of the same
|size|.

Motivation: the previous generation made hold_frac / relax_p1 depend on
|size| (packing-density extremity), scoring +0.176% farm-balanced with all
farms feasible -- but dei_n80_omnidir (dense, large-N, positive size) was
still below native (-0.054%) while parque_n10_omnidir (sparse, small-N,
negative size, similar |size|) is comfortably positive (+0.22%). Treating
both extremes identically via |size| is therefore *not* symmetric in effect:
a large-N dense farm has many more simultaneous spacing constraints to
resolve than a small-N sparse farm has boundary constraints, so it needs
disproportionately more unconstrained exploration time and a gentler
terminal feasibility push, not the same amount as its sparse mirror image.

Changes vs. the parent (all machinery otherwise identical):

  1. hold_frac now uses separate spans for size>0 (dense/large-N) and
     size<0 (sparse/small-N): the positive branch gets a larger span
     (0.06 -> 0.10) so dense farms get more exploration before constraints
     engage, while the negative branch keeps the parent's span (already
     tuned well for parque_n10).

  2. spike0 (terminal penalty spike) gets a *smaller* span on the positive
     branch (0.55 -> 0.35): a large-N dense farm has many turbines whose
     positions all get yanked simultaneously by a big late spike, which is
     more likely to overshoot into a worse feasible-but-suboptimal
     configuration than the same spike on a small-N sparse farm.

  3. relax_p1 (delay of full penalty engagement) gets more delay on the
     positive branch (span 0.10 -> 0.14) to pair with the longer hold.

  Negative-size (sparse) behavior is left exactly as in the parent since
  parque_n10_omnidir and parque_n20 already score well; only the
  dense/large-N branch is retuned.
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
_RELAX_P1_BASE = 0.45
_RELAX_P1_SPAN_POS = 0.14       # dense/large-N: delay full penalty more
_RELAX_P1_SPAN_NEG = 0.10       # sparse/small-N: parent value

_SPIKE_BASE = 2.0
_SPIKE_SPAN_POS = 0.35          # dense/large-N: gentler terminal spike
_SPIKE_SPAN_NEG = 0.55          # sparse/small-N: parent value
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

_HOLD_FRAC_BASE = 1.0 / 3.0
_HOLD_FRAC_SPAN_POS = 0.10      # dense/large-N: longer exploration plateau
_HOLD_FRAC_SPAN_NEG = 0.06      # sparse/small-N: parent value
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
    pos = max(size, 0.0)
    neg = max(-size, 0.0)

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = (_HOLD_FRAC_BASE
                 + _HOLD_FRAC_SPAN_POS * pos
                 + _HOLD_FRAC_SPAN_NEG * neg)
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
    spike_span = _SPIKE_SPAN_POS * pos + _SPIKE_SPAN_NEG * (-neg)
    spike0 = _SPIKE_BASE + spike_span                # dense farms -> milder late fix
    relax_p1 = (_RELAX_P1_BASE
                + _RELAX_P1_SPAN_POS * pos
                + _RELAX_P1_SPAN_NEG * neg)
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