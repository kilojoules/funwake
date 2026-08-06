"""Schedule: parent's graduated/spiked penalty, but relax bound, hold length,
warmup gentleness, and terminal spike are all *adapted* to farm geometry
(n_turbines, min_spacing/D packing ratio) instead of fixed constants tuned to
one farm size.

Rationale for the two weak parent farms:
  - parque_n10_omnidir (few turbines): the parent's fixed 0.45x relaxation and
    2.0x spike were tuned near n=50 and don't have to be right at n=10 or
    n=80. With few turbines there is more free room, so exploration can
    afford a gentler ramp-in (higher warm_lo) without early overlap risk.
  - dei_n80_omnidir (many turbines, denser layout): more turbines packed in
    a fixed-shape boundary means overlap constraints bind sooner/harder, so
    exploration needs a softer warm-in and a stronger terminal spike to claw
    back feasibility.

All shaping now derives from (n_turbines, min_spacing/D) around the training
farm's own geometry (n=50, ~native spacing ratio), so the schedule adapts
smoothly instead of re-hardcoding one farm's numbers.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_P0, _RELAX_P1 = 0.08, 0.45
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


def _ramp(p, p0, p1):
    """Traceable smoothstep from 0 at p0 to 1 at p1."""
    x = jnp.clip((p - p0) / max(p1 - p0, 1e-12), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)
    n_turb = float(n_turbines)
    spacing_ratio = float(min_spacing) / max(diam, 1e-9)   # packing tightness

    # geometry-adaptive shaping (all host-side floats, not traced)
    log_n = math.log(max(n_turb, 1.0) / 50.0)               # 0 at n=50 (training farm)

    hold_frac = _clip(1.0 / 3.0 + 0.05 * log_n, 0.22, 0.45)
    # more crowded (larger min_spacing/D, or more turbines) -> less relaxation
    relax_lo = _clip(0.45 + 0.08 * (spacing_ratio - 2.0) + 0.05 * log_n, 0.25, 0.70)
    spike_mag = _clip(2.0 + 0.5 * (spacing_ratio - 2.0) + 0.3 * log_n, 1.0, 3.5)
    # fewer turbines -> more room -> can start faster; more turbines -> gentler start
    warm_lo = _clip(0.30 - 0.10 * log_n, 0.15, 0.45)

    lr0 = _C * diam                                  # exploration lr from D
    n_hold = max(int(n_total * hold_frac), 1)
    n_decay = max(n_total - n_hold, 2)
    n_warm = max(int(n_total * _WARM_FRAC), 1)

    table = _decay_table(lr0, g_min, n_decay)

    s = jnp.asarray(step)
    k = jnp.clip(s - n_hold, 0, n_decay - 1)
    lr_base = jnp.take(table, k)                     # lr0 -> gamma_min

    warm = warm_lo + (1.0 - warm_lo) * jnp.clip(s / n_warm, 0.0, 1.0)
    lr = lr_base * warm

    p = s * (1.0 / n_total)                          # progress in [0, 1)
    relax = relax_lo + (1.0 - relax_lo) * _ramp(p, _RELAX_P0, _RELAX_P1)
    spike = 1.0 + spike_mag * _ramp(p, _SPIKE_P0, _SPIKE_P1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax * spike / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2