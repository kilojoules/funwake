"""Schedule: extend the threshold-gated high-N relief (introduced to help
dei_n80_omnidir without touching dei_n50/parque) from the alpha-floor alone
to also soften the terminal alpha spike and delay its engagement, all under
the SAME scale_pos > 0.5 gate.

Diagnosis vs. the current parent (gen 106, farm-balanced +0.1896%,
dei_n80_omnidir -0.0059%): the best-so-far branch (+0.1903%, worst farm
-0.0025%) already added a quadratic, gate-at-scale_pos>0.5 relief term to
alpha_denom_floor, which softened dei_n80_omnidir's terminal alpha coupling
without perturbing dei_n50 (scale_pos ~= 0.37, below the 0.5 gate) or the
parque farms (scale_pos == 0). dei_n80_omnidir is still the only
below-native farm, so the fix pushes further on the SAME lever set instead
of opening a new one that could destabilize the farms already positive:

  1. Raise _ALPHA_FLOOR_HIGH_N_SPAN 6.0 -> 8.0: more terminal-alpha relief
     for dei_n80_omnidir (high_n_gap ~= 0.21, so the extra floor term grows
     from ~0.26*g_min to ~0.35*g_min -- still zero for any farm below the
     0.5 gate).
  2. Add a matching quadratic dampener on spike0, gated identically, so the
     terminal feasibility spike itself is smaller for high-N farms (spike0
     already gets a *linear* scale_pos term via _SPIKE_SCALE_SPAN; this adds
     a second, gated term that only bites past the 0.5 threshold, giving
     dei_n80_omnidir specifically a gentler terminal ramp without changing
     dei_n50/parque's spike0 at all).
  3. Add a matching quadratic extension to relax_p1 (delay full penalty
     engagement further), gated the same way, so the highest-N farm holds
     its relaxed, exploration-friendly alpha a bit longer into the run
     before the terminal machinery engages.

All three additions are `high_n_gap**2 * const` terms that are exactly zero
whenever scale_pos <= 0.5 (i.e. n_turbines <~ 60), so dei_n50, parque_n10,
and parque_n20 are bit-for-bit unaffected -- only dei_n80_omnidir (and any
future, larger farm) sees a change. Everything else (inverse-time product
lr decay to gamma_min, short linear warmup, density-extremity relax/spike/
hold, absolute-N scale on hold/spike/relax_p1, the hold_frac-floored beta
ramp) is unchanged from the parent.
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

# Bound the terminal alpha denominator (lr_base floor) so alpha's 1/lr_base
# divergence is capped instead of fully re-engaging native's
# constraint-dominated tail on crowded (many-pair) or unusually-spaced
# layouts.
_ALPHA_FLOOR_SCALE_SPAN = 2.0
_ALPHA_FLOOR_EXT_SPAN = 0.3
# Threshold-gated quadratic relief for farms well past the linear regime
# (scale_pos > 0.5, i.e. N > 60) -- decouples dei_n80_omnidir's floor from
# dei_n50's without touching any farm below the gate.
_ALPHA_FLOOR_HIGH_N_THRESH = 0.5
_ALPHA_FLOOR_HIGH_N_SPAN = 8.0

# Same gate, softening the terminal spike and delaying penalty engagement
# further for the highest-N farms only (zero below the gate).
_SPIKE_HIGH_N_SPAN = 0.35
_RELAX_P1_HIGH_N_SPAN = 0.06

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
    high_n_gap = max(scale_pos - _ALPHA_FLOOR_HIGH_N_THRESH, 0.0)
    high_n_gap2 = high_n_gap * high_n_gap               # zero for scale_pos <= 0.5

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
              + _SPIKE_SCALE_SPAN * scale_pos
              - _SPIKE_HIGH_N_SPAN * high_n_gap2)      # extra terminal-spike relief past N~60
    relax_p1 = (_RELAX_P1_BASE + _RELAX_P1_SPAN * extremity
                + _RELAX_P1_SCALE_SPAN * scale_pos
                + _RELAX_P1_HIGH_N_SPAN * high_n_gap2)  # delay full penalty even further past N~60
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    # Cap the effective denominator at gamma_min*(1 + linear scale/extremity
    # terms + a threshold-gated quadratic term for scale_pos beyond 0.5) so
    # alpha's terminal divergence is bounded more for the largest-N (many-
    # pair) farms specifically, without perturbing farms below the gate.
    alpha_denom_floor = g_min * (1.0 + _ALPHA_FLOOR_SCALE_SPAN * scale_pos
                                  + _ALPHA_FLOOR_EXT_SPAN * extremity
                                  + _ALPHA_FLOOR_HIGH_N_SPAN * high_n_gap2)
    alpha_denom = jnp.maximum(lr_base, alpha_denom_floor)
    alpha = alpha0 * diam * relax * spike / alpha_denom

    # Never let the beta ramp start before the lr hold phase ends -- for
    # large-N/extreme-density farms, _BETA_P0_SCALE_SPAN pulls beta_p0 down
    # while _HOLD_FRAC_SPAN/_HOLD_FRAC_SCALE_SPAN push hold_frac up, and the
    # two can cross so momentum starts increasing during pure exploration
    # (before lr has decayed at all) on exactly the crowded, many-pair farms
    # that most need unaveraged gradient steps there.
    beta_p0 = max(_BETA_P0 + _BETA_P0_SCALE_SPAN * scale_pos, hold_frac)
    r = _ramp(p, beta_p0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2