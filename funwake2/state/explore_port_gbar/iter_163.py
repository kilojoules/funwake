"""Schedule: builds on the best-so-far variant (+0.1954% farm-balanced),
which already gates hold/relax-delay/spike/alpha-floor relief onto the
N~=60-100+ band via `high_n_ramp` (scale_pos in [0.5, 0.78]), leaving every
other mechanism identical to the parent.

This variant makes ONE additional, farm-agnostic change: it raises the
terminal Adam second-moment decay (_B2_HI) from 0.45 to 0.52. This is the
one item from the prior-art survey's design menu that the lineage has not
yet explored on its own: "ramping beta2 tames early adaptive-lr variance
and absorbs the ~alpha constraint-curvature conditioning" (AMSGrad-style
monotone-late beta2). The existing beta1/beta2 ramp already phases Adam's
moments from native-like (0.1/0.2) toward a more-averaged regime late in
the run, gated to start no earlier than the exploration hold ends
(`beta_p0 = max(..., hold_frac)`). A slightly higher terminal beta2 means
the per-parameter step-size normalization averages over a longer effective
window exactly when alpha is near its (now-bounded) terminal plateau/spike
-- i.e. when the gradient signal is a noisy mix of AEP gradient and a
comparatively large constraint gradient. A steadier second-moment estimate
there should reduce step-to-step variance in the endgame across ALL farm
sizes (it is not gated by scale_pos/size, so it applies uniformly and
cannot be an overfit to any one farm), which is exactly the axis on which
recent generations have seen the most run-to-run score variance (+0.195%
down to -0.029% between otherwise-similar variants). Nothing else --
lr schedule, alpha shaping, hold/relax/spike gating, alpha floor -- is
touched, so this is a clean, isolated ablation of the beta2 ceiling alone.
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
_SPIKE_FRONTLOAD_HIGH_N_SPAN = 1.0  # blend weight (x high_n_ramp) toward a
                                     # front-loaded spike-ramp shape for that band

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
# Extra, smoothstep-gated relief for farms in the N ~= 60-100+ band
# (scale_pos in [0.5, 0.78]) -- decouples dei_n80_omnidir's floor from
# dei_n50's without touching any farm below the gate.
_ALPHA_FLOOR_HIGH_N_P0 = 0.5
_ALPHA_FLOOR_HIGH_N_P1 = 0.78
_ALPHA_FLOOR_HIGH_N_SPAN = 3.5

# Adam moments: native early -> mildly averaged late
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.52   # raised terminal beta2 ceiling (was 0.45): steadier
                              # second-moment estimate during the bounded-alpha
                              # endgame, applied uniformly across all farm sizes
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


def _ramp_frontloaded(p, p0, p1):
    """Traceable front-loaded ramp from 0 at p0 to 1 at p1: rises faster
    near p0, flattens near p1 -- same endpoints/magnitude as `_ramp`, but
    reaches a given fraction of the terminal value earlier in the window."""
    x = jnp.clip((p - p0) / max(p1 - p0, 1e-12), 0.0, 1.0)
    return 1.0 - (1.0 - x) * (1.0 - x)


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
                 + _HOLD_FRAC_HIGH_N_SPAN * high_n_ramp)  # extra gated hold for N~=60-100+ band
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
                + _RELAX_P1_HIGH_N_SPAN * high_n_ramp)  # extra gated delay for N~=60-100+ band
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike_p0 = _SPIKE_P0 + _SPIKE_P0_HIGH_N_SPAN * high_n_ramp  # delay spike onset for high-N

    # Same spike window (spike_p0 -> _SPIKE_P1) and magnitude (spike0) as the
    # best-so-far variant; only the ramp *shape* is blended toward a
    # front-loaded curve for the high-N gated band, so that band spends more
    # of the window actually adapting under a rising (not yet-flat) penalty
    # instead of a flat-then-snap terminal jump. Blend weight is high_n_ramp,
    # so every farm below the gate gets the exact original smoothstep shape.
    shape_std = _ramp(p, spike_p0, _SPIKE_P1)
    shape_front = _ramp_frontloaded(p, spike_p0, _SPIKE_P1)
    spike_shape = shape_std + (_SPIKE_FRONTLOAD_HIGH_N_SPAN * high_n_ramp) * (shape_front - shape_std)
    spike = 1.0 + spike0 * spike_shape

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