"""DIAGNOSTIC ONLY (item-1 reconciliation) — native TopFarm schedule forced to
lr0 = 50 (the v1 under-tuned default the old parqo_native_ms baseline used),
run through skeleton_v2's scale-aware machinery. If skeleton_v2 + this schedule
reproduces parqo_native_ms/uniform|n30 per-seed, the multizone machinery is
faithful and the c*D 0/10 is purely the lr0 choice. NOT a seed / not deployed.
"""
import os
import sys

import jax.numpy as jnp

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_pix = os.path.join(_ROOT, "dependencies", "pixwake", "src")
if _pix not in sys.path:
    sys.path.insert(0, _pix)
from pixwake.optim.sgd import _compute_mid_bisection  # noqa: E402

_LR0 = 50.0                      # v1 under-tuned default (old parqo baseline)
_BISECT_LOWER = 0.0
_BISECT_UPPER = 0.1


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    lr0 = _LR0                                # fixed lr0=50 (NOT c*D)
    n_const = int(total_steps) // 3
    max_iter = int(total_steps) - n_const
    mid = _compute_mid_bisection(
        learning_rate=float(lr0), gamma_min=float(gamma_min),
        max_iter=max_iter, lower=_BISECT_LOWER, upper=_BISECT_UPPER)
    ts = jnp.arange(1, max_iter + 1)
    k = jnp.maximum(step - n_const, 0)
    factors = jnp.where(ts <= k, 1.0 / (1.0 + mid * ts), 1.0)
    lr = lr0 * jnp.prod(factors)
    alpha = alpha0 * float(D) / jnp.maximum(lr, 1e-30)   # = mean|grad J|/lr
    return lr, alpha, 0.1, 0.2
