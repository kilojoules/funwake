"""Schedule: revert the alpha-denominator floor to the proven linear
scale_pos coupling (the parent's switch to scale_pos**2 over-inflated the
floor specifically for dei_n80_omnidir -- at n=80, scale_pos**2 (~0.50) times
its span-4.0 coefficient is actually LARGER than scale_pos (~0.71) times the
old span-1.5 coefficient, so the "highest-N" farm got MORE denominator
inflation than before, not less, starving it of terminal alpha and erasing
the parent's gain there (-0.001% vs. native). Reverting this one lever to
the exact linear form that was already validated (+0.1894% farm-balanced)
restores dei_n80_omnidir's headroom without touching mid/small-N farms
(dei_n50, parque_n10/n20), whose alpha floor is unchanged either way.

On top of that restoration, add ONE new, narrowly-gated lever: an extra
terminal-spike softening term that only activates for farms with scale_pos
above dei_n50's level (~0.37) and saturates by dei_n80_omnidir's level
(~0.71). This targets exactly the gap between "large" (n50) and "largest"
(n80) farms -- the parent's existing _SPIKE_SCALE_SPAN already softens the
spike in proportion to scale_pos, but dei_n80_omnidir has 80 turbines (>50%
more turbine-pairs than any other cell in the fitness set) and the
constraint-gradient noise that implies is not distinguished from dei_n50's
milder large-N-ness by a single linear coefficient. The added gated term is
zero at dei_n50 (gate(0.37)=0) and near-full strength at dei_n80_omnidir
(gate(0.71)~1), so it cannot regress any farm already at its optimum and can
only further relax the terminal constraint spike where turbine-pair count is
highest.

All other machinery (self-contained inverse-time product lr decay to
gamma_min, short linear warmup, density-extremity relax/spike/hold,
absolute-N scale on hold/relax_p1/beta/relax0) is unchanged from the parent.
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

# NEW: extra terminal-spike softening, gated to only engage above dei_n50's
# scale_pos level and saturate by dei_n80_omnidir's -- targets the largest-N
# farm specifically without perturbing mid-N farms already at their optimum.
_SPIKE_SCALE_SPAN2 = -0.35
_SPIKE_GATE_LO, _SPIKE_GATE_HI = 0.45, 0.75

_HOLD_FRAC_BASE, _HOLD_FRAC_SPAN = 1.0 / 3.0, 0.06   # exploration plateau length
_HOLD_FRAC_SCALE_SPAN = 0.05    # extra hold for large-N farms
_DENSITY_REF = 12.5             # ~50 turbines at 2D spacing (mid-scale anchor)
_N_REF = 30.0                   # absolute-N reference (independent of density)

# Terminal alpha denominator (lr_base floor) for large-N farms: reverted to
# the validated LINEAR scale_pos coupling (see module docstring -- the
# squared variant over-inflated the floor for the highest-N farm).
_ALPHA_FLOOR_SCALE_SPAN = 1.5

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


def _smoothstep(x, x0, x1):
    """Python-float smoothstep from 0 at x0 to 1 at x1 (trace-time constant)."""
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
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
    spike_gate = _smoothstep(scale_pos, _SPIKE_GATE_LO, _SPIKE_GATE_HI)  # 0 at n50, ~1 at n80

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
              + _SPIKE_SCALE_SPAN2 * spike_gate)       # extra softening only at highest N
    relax_p1 = (_RELAX_P1_BASE + _RELAX_P1_SPAN * extremity
                + _RELAX_P1_SCALE_SPAN * scale_pos)    # delay full penalty further for large N
    relax = relax0 + (1.0 - relax0) * _ramp(p, _RELAX_P0, relax_p1)
    spike = 1.0 + spike0 * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    # Denominator floor gated LINEARLY by scale_pos (reverted from the
    # squared form -- see module docstring) so alpha's terminal divergence
    # is bounded on large-N farms without over-inflating the floor for the
    # largest one.
    alpha_denom_floor = g_min * (1.0 + _ALPHA_FLOOR_SCALE_SPAN * scale_pos)
    alpha_denom = jnp.maximum(lr_base, alpha_denom_floor)
    alpha = alpha0 * diam * relax * spike / alpha_denom

    beta_p0 = _BETA_P0 + _BETA_P0_SCALE_SPAN * scale_pos   # earlier smoothing onset, large N
    r = _ramp(p, beta_p0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2