"""Schedule: farm-adaptive graduated penalty on top of native decay.

Diagnosis of the parent (gen 4, farm-balanced +0.0157%): all farms stayed
feasible, so the parent's terminal alpha spike (x3 native) has slack to
spare everywhere. The two farms it loses AEP on -- parque_n10_omnidir and
dei_n80_omnidir -- are opposite ends of the turbine-count range (10 vs 80)
but share an omnidirectional wind rose, i.e. no preferred streaking
direction to exploit. That argues for two independent, *continuous*
adaptations (no branching -- n_turbines/D/min_spacing/total_steps are
static Python floats at trace time, so ordinary arithmetic on them is
fine; only `step` and `alpha0` stay inside jnp):

  1. Soften the terminal spike globally (2.0x -> ~1.4x native) and push it
     later/narrower, since feasibility already had margin everywhere --
     recovering AEP is free where the constraint was never tight.
  2. Scale the *depth* of the relaxation and its restoration point with
     n_turbines relative to a reference of 50 (the farm the parent is
     tuned closest to): sparse farms (n=10) get a deeper, longer
     relaxation window (more room to explore before constraints bind),
     crowded farms (n=80) get a shallower, earlier-restoring one (more
     conflicts to resolve, less time to spare). Same idea nudges the
     lr hold fraction and the exploration lr scale slightly.

Everything else (log-gamma product-decay table, warmup, beta phase
transition) is unchanged from the parent -- it wasn't implicated by the
per-farm breakdown.
"""

import math

import jax.numpy as jnp

# exploration scale: lr0 = c * D, c = 200 m / 240 m (DEI-fitted diameter rule)
_C = 200.0 / 240.0

_WARM_FRAC = 0.015              # linear warmup inside the plateau
_WARM_LO = 0.30                 # lr multiplier at step 0

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
    return min(max(x, lo), hi)


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = int(total_steps)
    diam = float(D)
    g_min = float(gamma_min)
    n_t = float(n_turbines)

    # farm-count ratio vs. the reference (dei_n50) the base schedule is
    # closest to; <1 sparse (e.g. n=10), >1 crowded (e.g. n=80).
    n_scale = n_t / 50.0

    hold_frac = _clip(1.0 / 3.0 + 0.03 * (n_scale - 1.0), 0.25, 0.42)

    # graduated-penalty depth/window: sparse farms relax deeper and longer
    # (more slack to explore before constraints bind), crowded farms relax
    # shallower and restore sooner (more conflicts, less slack).
    relax = _clip(0.38 + 0.10 * (n_scale - 1.0), 0.25, 0.55)
    relax_p0 = _clip(0.08 + 0.03 * (n_scale - 1.0), 0.04, 0.16)
    relax_p1 = _clip(0.50 - 0.05 * (n_scale - 1.0), 0.35, 0.55)

    # terminal feasibility spike: softened overall vs. native x3 (every
    # farm stayed feasible with margin to spare), still scaled up a bit
    # for crowded farms which need more late-stage enforcement.
    spike_mag = _clip(1.4 + 0.3 * (n_scale - 1.0), 1.0, 2.2)
    spike_p0, spike_p1 = 0.92, 1.00

    lr0 = _C * diam                                  # exploration lr from D
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
    relax_shape = relax + (1.0 - relax) * _ramp(p, relax_p0, relax_p1)
    spike_shape = 1.0 + spike_mag * _ramp(p, spike_p0, spike_p1)

    # native coupling recovered via /D: alpha0 = mean|grad J|/D, so
    # alpha0*D/lr_base = mean|grad J|/lr_base, then shaped.  Couple to the
    # un-warmed lr so the warmup does not disturb the penalty profile.
    alpha = alpha0 * diam * relax_shape * spike_shape / jnp.maximum(lr_base, 1e-30)

    r = _ramp(p, _BETA_P0, _BETA_P1)
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2