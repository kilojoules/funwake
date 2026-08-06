"""Schedule: asymmetric dense-side relief on top of the symmetric-extremity
parent, plus reverting to the uncapped native alpha coupling.

The parent added a bounded alpha plateau (soft cap on alpha0*D*relax*spike/
lr_base) on top of the prior best (uncapped) schedule, and that regressed
the farm-balanced mean (+0.1247% vs the prior best's +0.1762%), with
dei_n80_omnidir -- the largest-N, most densely-packed farm in the suite --
now scoring *below* native (-0.028%). A cap that saturates the penalty is
exactly the wrong fix for a dense farm that needs strong late-stage
constraint enforcement to reach feasibility without giving back AEP, so:

  1. Drop the alpha cap entirely -- go back to the uncapped native coupling
     alpha0*D*relax*spike/lr_base (this alone was already the better of the
     two known configurations).

  2. The prior best treated packing-density extremity *symmetrically*: both
     the dense end (dei_n80_omnidir) and the sparse end (parque_n10_omnidir)
     got the same longer hold/delayed-relax treatment. But only the dense
     end is underperforming (sparse already beats native comfortably) --
     symmetric tuning wastes headroom on an end that didn't need help and
     under-serves the end that does. Split the extremity-dependent terms
     (hold_frac, relax0 span, relax engagement delay, terminal spike span)
     into separate dense-side / sparse-side coefficients, and only push the
     dense-side coefficients further: longer AEP-only exploration hold,
     more early constraint slack (higher relax0), a later relax engagement
     window, and a stronger terminal feasibility spike. The sparse-side
     coefficients are left exactly as in the prior best (unchanged, since
     that end already works). Mid-density farms (extremity ~ 0) are
     unaffected by either branch.

  All other machinery (self-contained inverse-time product lr decay to
  gamma_min, short linear warmup, beta1/beta2 phase-transition) is
  unchanged from the parent.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_BASE = 0.45
_RELAX_SPAN_DENSE, _RELAX_SPAN_SPARSE = 0.16, 0.11   # relax0 = base + span * size
_RELAX_P0 = 0.08
_RELAX_P1_BASE = 0.45
_RELAX_P1_SPAN_DENSE, _RELAX_P1_SPAN_SPARSE = 0.16, 0.10   # engagement-window delay
_SPIKE_BASE = 2.0
_SPIKE_SPAN_DENSE, _SPIKE_SPAN_SPARSE = 0.70, 0.55          # spike0 = base + span * size
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

_HOLD_FRAC_BASE = 1.0 / 3.0
_HOLD_FRAC_SPAN_DENSE, _HOLD_FRAC_SPAN_SPARSE = 0.12, 0.06   # exploration plateau length
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
    dense = size > 0.0                                  # dense/large-N side vs sparse/small-N side

    hold_span = _HOLD_FRAC_SPAN_DENSE if dense else _HOLD_FRAC_SPAN_SPARSE
    relax_span = _RELAX_SPAN_DENSE if dense else _RELAX_SPAN_SPARSE
    relax_p1_span = _RELAX_P1_SPAN_DENSE if dense else _RELAX_P1_SPAN_SPARSE
    spike_span = _SPIKE_SPAN_DENSE if dense else _SPIKE_SPAN_SPARSE

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = _HOLD_FRAC_BASE + hold_span * extremity   # longer plateau for dense farms
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

    relax0 = _RELAX_BASE + relax_span * size          # sparse farms -> freer early
    spike0 = _SPIKE_BASE + spike_span * size          # dense farms -> stronger late fix
    relax_p1 = _RELAX_P1_BASE + relax_p1_span * extremity   # extreme farms: delay full penalty
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    # Uncapped: the bounded plateau tried in the parent regressed the dense,
    # large-N farm, so the native diverging-toward-gamma_min coupling is kept.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2