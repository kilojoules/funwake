"""Schedule: crowding-adaptive graduated / terminally-spiked penalty.

Changes vs. the parent (native TopFarm-SGD coupling with fixed graduated
penalty):
  1. same self-contained inverse-time product decay (log-gamma bisection,
     cached lookup table) reaching gamma_min exactly on the final step, and
     the same short linear warmup inside the constant-lr exploration
     plateau;
  2. the graduated-penalty shape (how relaxed exploration is, how strong the
     terminal feasibility spike is, and when that spike starts) is no longer
     a fixed constant -- it is a function of a *crowding* proxy
     `n_turbines * (min_spacing / D)^2`, which estimates how tightly the
     layout must pack turbines relative to the reference training farm
     (dei_n50, ~50 turbines at ~3D spacing). Sparse/low-N farms (few
     turbines, generous spacing) get a shallower relaxation and a smaller,
     earlier-settling terminal spike so the correction doesn't overshoot the
     optimum in the little room those layouts have to move; dense/high-N
     farms get a slightly stronger, earlier-starting spike so there are
     enough remaining steps to resolve the extra spacing conflicts;
  3. Adam moments unchanged: native (0.1, 0.2) during exploration ->
     mildly-averaged (0.25, 0.45) during the feasibility/refinement phase.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_HOLD_FRAC = 1.0 / 3.0          # constant-lr exploration plateau
_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

# penalty shaping, multiplying the native alpha = mean|grad J| / lr coupling
_RELAX_P0, _RELAX_P1 = 0.08, 0.45

# Adam moments: native early -> mildly averaged late
_B1_LO, _B1_HI = 0.1, 0.25
_B2_LO, _B2_HI = 0.2, 0.45
_BETA_P0, _BETA_P1 = 0.50, 0.90

# crowding reference: dei_n50 (~50 turbines, ~3D min spacing)
_CROWD_REF = 50.0 * 3.0 ** 2

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


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)
    n_turb = float(n_turbines)
    spacing_ratio = float(min_spacing) / max(diam, 1e-12)

    # crowding proxy: how many "min-spacing disks" the layout must pack,
    # relative to the reference training farm (dei_n50).
    crowd = n_turb * spacing_ratio * spacing_ratio
    crowd_norm = _clip(crowd / _CROWD_REF, 0.15, 6.0)
    log_crowd = math.log(crowd_norm)

    # sparse/low-N farms -> shallower relaxation (less slack needed);
    # dense/high-N farms -> deeper relaxation (more room to spread first).
    relax = _clip(0.45 * (crowd_norm ** 0.12), 0.32, 0.58)

    # sparse farms -> smaller, later terminal spike (little room to overshoot
    # in); dense farms -> stronger, earlier-starting spike (more time to
    # settle the extra spacing conflicts it creates).
    spike_mag = _clip(2.0 * (crowd_norm ** 0.22), 1.2, 3.2)
    spike_p0 = _clip(0.90 - 0.04 * log_crowd, 0.82, 0.93)
    spike_p1 = min(spike_p0 + 0.10, 1.0)

    lr0 = _C * diam                                  # exploration lr from D
    n_hold = max(int(n_total * _HOLD_FRAC), 1)
    n_decay = max(n_total - n_hold, 2)
    n_warm = max(int(n_total * _WARM_FRAC), 1)

    table = _decay_table(lr0, g_min, n_decay)

    s = jnp.asarray(step)
    k = jnp.clip(s - n_hold, 0, n_decay - 1)
    lr_base = jnp.take(table, k)                     # lr0 -> gamma_min

    warm = _WARM_LO + (1.0 - _WARM_LO) * jnp.clip(s / n_warm, 0.0, 1.0)
    lr = lr_base * warm

    p = s * (1.0 / n_total)                          # progress in [0, 1)
    relax_curve = relax + (1.0 - relax) * _ramp(p, _RELAX_P0, _RELAX_P1)
    spike_curve = 1.0 + spike_mag * _ramp(p, spike_p0, spike_p1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax_curve * spike_curve / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2