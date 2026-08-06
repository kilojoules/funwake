"""Schedule: synchronize a terminal beta2 (second-moment) boost with the
existing alpha terminal spike window (isolated single-lever change).

Diagnosis vs. the parent: the parent already ramps alpha's terminal
"spike" multiplier (1 + spike0 * ramp(p, 0.90, 1.00)) to restore
feasibility in the last ~10% of steps, on every farm (spike0's magnitude
varies with density/size and scale_pos, but the *window* p in [0.90,1.00]
fires for all farms). That spike introduces a sudden, farm-dependent jump
in the constraint-gradient magnitude right when beta1/beta2 have already
mostly finished their own independent ramp (beta_p0..0.90 -> _BETA_P1=1.00,
i.e. beta1/beta2 are also still transitioning across that exact window,
but their ramp target (_B2_HI=0.45) was tuned only against the *shape* of
the alpha ramp, not against the amount of extra gradient volatility the
spike itself injects). Per the prior-art guidance's "phase-transition the
Adam moments with the alpha phase" bet (already partially embodied via
beta_p0/_BETA_P1, but not keyed to the spike magnitude itself): reusing
the exact same terminal ramp fraction that drives the alpha spike to also
add a small extra beta2 boost (toward a higher ceiling than the base
ramp's _B2_HI) gives the second-moment estimate more averaging exactly
when -- and only when, and only as much as -- the constraint gradient
itself is spiking, which should damp step-to-step overshoot from the
spike without touching alpha's magnitude, lr, or beta1. Farms with a
small spike0 (mid-density) get a small beta2 boost; farms with a large
spike0 (density extremes / large-N) get more averaging exactly where they
need it most. At p < 0.90 this term is exactly zero, so it cannot change
behavior anywhere the parent already scores well.

All other machinery (self-contained inverse-time product lr decay to
gamma_min, short linear warmup, density-extremity relax/spike/hold,
absolute-N scale on hold/spike/relax_p1/beta/relax0, the alpha terminal
denominator floor for large-N farms) is unchanged from the parent.
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
_SPIKE_BASE, _SPIKE_SPAN = 2.0, 0.55    # spike0 = base + span * size
_SPIKE_SCALE_SPAN = -0.4        # soften terminal spike for large-N farms
_SPIKE_P0, _SPIKE_P1 = 0.90, 1.00

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.06   # exploration plateau length
_HOLD_FRAC_SCALE_SPAN = 0.05    # extra hold for large-N farms
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)
_N_REF = 30.0                   # absolute-N reference (independent of density)

# bound the terminal alpha denominator (lr_base floor) for large-N farms
# so alpha's 1/lr_base divergence is capped instead of fully re-engaging
# native's constraint-dominated tail on crowded (many-pair) layouts.
_ALPHA_FLOOR_SCALE_SPAN = 1.5

# Adam moments: native early -> mildly averaged late
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.45
_BETA_P0, _BETA_P1 = 0.50, 0.90
_BETA_P0_SCALE_SPAN = -0.10     # earlier averaging onset for large-N farms

# NEW: extra beta2 averaging synced to the alpha terminal spike window,
# scaled by the same spike0 magnitude that drives the alpha spike itself.
_B2_SPIKE_GAIN = 0.5            # fraction of spike0 converted into extra beta2 ceiling
_B2_SPIKE_CAP = 0.20            # max extra beta2 headroom above _B2_HI

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

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = (_HOLD_FRAC_BASE + _HOLD_FRAC_SPAN * extremity
                 + _HOLD_FRAC_SCALE_SPAN * scale_pos)   # longer plateau at extremes AND large N
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
              + _SPIKE_SCALE_SPAN * scale_pos)        # softened terminal spike for large N
    relax_p1 = (_RELAX_P1_BASE + _RELAX_P1_SPAN * extremity
                + _RELAX_P1_SCALE_SPAN * scale_pos)    # delay full penalty further for large N
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike_ramp = _ramp(p, _SPIKE_P0, _SPIKE_P1)         # shared terminal-spike phase signal
    spike = 1.0 + spike0 * spike_ramp

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    # cap the effective denominator at gamma_min*(1+k*scale_pos) instead
    # of letting it ride lr_base all the way to gamma_min, so alpha's
    # terminal divergence is bounded on large-N (many-pair) farms only.
    alpha_denom_floor = g_min * (1.0 + _ALPHA_FLOOR_SCALE_SPAN * scale_pos)
    alpha_denom = jnp.maximum(lr_base, alpha_denom_floor)
    alpha = alpha0 * diam * relax * spike / alpha_denom

    beta_p0 = _BETA_P0 + _BETA_P0_SCALE_SPAN * scale_pos   # earlier smoothing onset, large N
    r = _ramp(p, beta_p0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    # NEW: extra beta2 averaging synced to the SAME terminal-spike phase
    # that drives alpha's spike, scaled by spike0 itself -- more headroom
    # exactly on the farms/steps where the constraint gradient is jumping
    # hardest, none where it isn't (spike_ramp == 0 for p < 0.90).
    b2_extra = jnp.clip(_B2_SPIKE_GAIN * spike0, 0.0, _B2_SPIKE_CAP)
    beta2 = jnp.clip(beta2 + b2_extra * spike_ramp, 0.0, 0.999)

    return lr, alpha, beta1, beta2