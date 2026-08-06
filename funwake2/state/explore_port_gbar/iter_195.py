"""Schedule: reverts to the proven `mid_gate` (extremity-gated) mid-N relief
mechanism -- which scored +0.2007% farm-balanced -- in place of the
scale_pos-parabola `mid_n_bump` variant that regressed to +0.1959%.

dei_n50 and dei_n50_uniform share IDENTICAL n_turbines/D/min_spacing, so
`size`/`extremity`/`scale_pos` are bit-for-bit identical between them: no
feature available to schedule_fn can tell them apart. Both n50 farms must
therefore be boosted together. The extremity-gated mid_gate approach did
this better than the scale_pos-gated bump, so it's restored as the base.
On top of that proven base, three small, conservative extensions:
  1. Slightly larger mid_gate relief magnitudes (hold/relax-delay/spike),
     since the worst farm (dei_n50_uniform) still trails n80/parque.
  2. A modest alpha-floor relief term gated on mid_gate, mirroring the
     high_n_ramp band's (already-validated) alpha floor relief -- the
     mid_gate band never got this lever despite the same rationale
     (crowded, many-pair layout needs bounded terminal alpha).
  3. mid_gate's fade window widened slightly (extremity 0->0.34 instead of
     0->0.30) so farms just past the old cutoff still get partial relief.
All changes are strictly additive/gated so parque_n* (low N, extremity~0
gate value near 1 anyway) and dei_n80_omnidir (high_n_ramp~1, mid_gate~0)
are essentially unaffected.
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
# Extra, smoothstep-gated relief for farms in the N ~= 60-100+ band
# (scale_pos in [0.5, 0.78]) -- decouples dei_n80_omnidir's floor from
# dei_n50's without touching any farm below the gate.
_ALPHA_FLOOR_HIGH_N_P0 = 0.5
_ALPHA_FLOOR_HIGH_N_P1 = 0.78
_ALPHA_FLOOR_HIGH_N_SPAN = 3.5

# Gate for near-reference-density farms (extremity ~= 0), which get the
# LEAST relax delay / hold extension / spike softening of any band --
# despite dei_n50 farms being the global worst-scoring. Fades to 0 by
# extremity=0.34 and is damped wherever high_n_ramp is already active, so
# dei_n80/parque are unaffected.
_MID_GATE_P0, _MID_GATE_P1 = 0.0, 0.34
_MID_HOLD_FRAC_SPAN = 0.028      # extra exploration hold
_MID_RELAX_P1_SPAN = 0.065       # extra delay of constraint engagement
_MID_SPIKE_SPAN = -0.32          # softening of the terminal spike
_MID_ALPHA_FLOOR_SPAN = 1.1      # small extra bound-relief on terminal alpha

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

    # near-reference-density gate: 1 at extremity=0, fades to 0 by
    # extremity=0.34, and is damped wherever the high-N band is already
    # active (avoids double-dipping on farms that are both near-reference
    # density AND large-N, which shouldn't occur but keeps the gate honest).
    mid_gate = (1.0 - _smoothstep_scalar(extremity, _MID_GATE_P0, _MID_GATE_P1)) * (1.0 - high_n_ramp)

    lr0 = _C * diam                                  # exploration lr from D
    hold_frac = (_HOLD_FRAC_BASE + _HOLD_FRAC_SPAN * extremity
                 + _HOLD_FRAC_SCALE_SPAN * scale_pos     # longer plateau at extremes AND large N
                 + _HOLD_FRAC_HIGH_N_SPAN * high_n_ramp   # extra gated hold for N~=60-100+ band
                 + _MID_HOLD_FRAC_SPAN * mid_gate)        # extra gated hold near-reference density
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
              + _SPIKE_HIGH_N_SPAN * high_n_ramp       # extra gated softening for high-N band
              + _MID_SPIKE_SPAN * mid_gate)            # small gated softening near-reference density
    relax_p1 = (_RELAX_P1_BASE + _RELAX_P1_SPAN * extremity
                + _RELAX_P1_SCALE_SPAN * scale_pos      # further delay for large-N farms
                + _RELAX_P1_HIGH_N_SPAN * high_n_ramp   # extra gated delay for N~=60-100+ band
                + _MID_RELAX_P1_SPAN * mid_gate)        # extra gated delay near-reference density
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike_p0 = _SPIKE_P0 + _SPIKE_P0_HIGH_N_SPAN * high_n_ramp  # delay spike onset for high-N
    spike = 1.0 + spike0 * _ramp(p, spike_p0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha_denom_floor = g_min * (1.0 + _ALPHA_FLOOR_SCALE_SPAN * scale_pos
                                  + _ALPHA_FLOOR_EXT_SPAN * extremity
                                  + _ALPHA_FLOOR_HIGH_N_SPAN * high_n_ramp
                                  + _MID_ALPHA_FLOOR_SPAN * mid_gate)
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