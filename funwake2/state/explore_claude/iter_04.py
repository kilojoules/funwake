import math

import jax.numpy as jnp

# diameter-rule constant: c = (DEI best lr0) / (DEI D) = 200 / 240
_C = 200.0 / 240.0
_PEAK_BOOST = 1.18     # push the exploration peak ~18% above the pure diameter rule (parent: 1.12)
_CONST_FRAC = 0.45     # hold the peak lr longer (parent held 0.40 of steps)
_SPIKE_START = 0.93    # terminal feasibility spike starts slightly earlier (parent: 0.94)
_SPIKE_WIDTH = 0.015
_SPIKE_GAIN = 4.0      # stronger terminal alpha ramp to cover the hotter exploration (parent: 3.0)
_BISECT_LOWER = 0.0
_BISECT_UPPER = 0.1


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total = int(total_steps)
    lr0 = _PEAK_BOOST * _C * float(D)           # exploration lr built from D
    n_const = int(round(_CONST_FRAC * total))   # constant-lr exploration phase
    max_iter = max(1, total - n_const)          # compounding decay horizon

    # Trace-time bisection on Python floats: find mid so the compounding
    # product decay lands exactly on gamma_min after max_iter decay steps.
    # Shorter decay horizon + higher peak both fold into log_target, so the
    # cool-down still terminates exactly at the metre-valued tolerance.
    log_target = math.log(float(lr0)) - math.log(float(gamma_min))

    def _decay_log_sum(mid):
        return sum(math.log1p(mid * t) for t in range(1, max_iter + 1))

    lo, hi = _BISECT_LOWER, _BISECT_UPPER
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if _decay_log_sum(m) < log_target:
            lo = m
        else:
            hi = m
    mid = 0.5 * (lo + hi)

    ts = jnp.arange(1, max_iter + 1)                   # 1 .. max_iter
    k = jnp.maximum(step - n_const, 0)                 # decay steps applied so far
    factors = jnp.where(ts <= k, 1.0 / (1.0 + mid * ts), 1.0)
    lr = lr0 * jnp.prod(factors)                       # peak hold, then decay -> gamma_min

    # native coupling recovered via /D: alpha0 = mean|grad J|/D  ==>
    #   alpha = alpha0 * D / lr = mean|grad J| / lr
    alpha = alpha0 * float(D) / jnp.maximum(lr, 1e-30)

    # terminal feasibility-restoration spike: smooth logistic ramp multiplying
    # alpha by up to (1 + _SPIKE_GAIN) in the tail, where lr is already near
    # gamma_min — negligible AEP cost. Started earlier and boosted vs the
    # parent so the hotter, longer exploration phase still finishes strictly
    # feasible on boundary and spacing.
    frac = step / total
    spike = _SPIKE_GAIN / (1.0 + jnp.exp(-(frac - _SPIKE_START) / _SPIKE_WIDTH))
    alpha = alpha * (1.0 + spike)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2