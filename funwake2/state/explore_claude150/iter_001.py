import os
import sys

import jax.numpy as jnp

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pix = os.path.join(_ROOT, "dependencies", "pixwake", "src")
if _pix not in sys.path:
    sys.path.insert(0, _pix)
from pixwake.optim.sgd import _compute_mid_bisection  # noqa: E402

# Diameter-rule constant, nudged up ~15% from the gen-1 value (200/240 ≈ 0.833):
# a slightly hotter exploration peak, held longer (one-cycle guidance), with the
# terminal feasibility restoration strengthened to compensate.
_C = 230.0 / 240.0
_CONST_NUM, _CONST_DEN = 2, 5     # constant-lr phase: first 40% (was 1/3)
_WARM_DEN = 50                    # linear lr warmup over the first ~2% of steps
_BISECT_LOWER = 0.0
_BISECT_UPPER = 0.1
_SPIKE_GAIN = 4.0                 # terminal alpha spike: up to ~5x native coupling
_SPIKE_CENTER = 0.95              # spike engages over the final ~5% of steps
_SPIKE_WIDTH = 0.012


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    lr0 = _C * float(D)                      # exploration lr scale from D
    n_total = int(total_steps)
    n_const = n_total * _CONST_NUM // _CONST_DEN   # longer high-lr plateau
    n_warm = max(n_total // _WARM_DEN, 1)
    max_iter = n_total - n_const             # compounding decay horizon

    # mid depends only on (lr0, gamma_min, max_iter) — all Python floats at
    # trace time, so this is the same precomputed bisection pixwake performs.
    # The shorter horizon yields a steeper decay that still lands on gamma_min.
    mid = _compute_mid_bisection(
        learning_rate=float(lr0), gamma_min=float(gamma_min),
        max_iter=max_iter, lower=_BISECT_LOWER, upper=_BISECT_UPPER)

    ts = jnp.arange(1, max_iter + 1)                   # 1 .. max_iter
    k = jnp.maximum(step - n_const, 0)                 # decay steps applied
    factors = jnp.where(ts <= k, 1.0 / (1.0 + mid * ts), 1.0)
    lr_decay = lr0 * jnp.prod(factors)                 # envelope: lr0 -> gamma_min

    # Short linear warmup damps the first few steps of the hotter peak so the
    # feasible grid init is not blown apart; it scales lr only, not alpha.
    warm = jnp.minimum((step + 1.0) / n_warm, 1.0)
    lr = lr_decay * warm

    # Native coupling recovered via /D on the decay ENVELOPE (not the warmed lr):
    #   alpha = alpha0 * D / lr_decay = mean|grad J| / lr_decay
    # so alpha still diverges as lr -> gamma_min, guaranteeing late feasibility.
    alpha = alpha0 * float(D) / jnp.maximum(lr_decay, 1e-30)

    # Terminal feasibility-restoration spike (filter/funnel idea): a smooth
    # logistic boost of alpha over the final ~5% of steps, adding margin for
    # the more aggressive exploration phase. Fully traceable in step.
    frac = (step + 1.0) / n_total
    spike = 1.0 + _SPIKE_GAIN / (1.0 + jnp.exp(-(frac - _SPIKE_CENTER) / _SPIKE_WIDTH))
    alpha = alpha * spike

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2