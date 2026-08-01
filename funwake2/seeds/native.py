"""Native TopFarm monotone schedule, ported to the FunWake-2 scale-aware
signature (seeded ancestor: ancestor=native, port_transform=lr0->c*D, alpha->/D).

Reproduces pixwake ``topfarm_sgd_solve`` / ``results/lr0_tuning/baseline_schedule.py``
(the object the v1 diameter-rule analysis measured), but with:

  * **internal lr scale built from D** — ``lr0 = c * D`` with
    ``c = 200/240 = 0.8333...`` (the diameter-rule constant, DEI best lr0 / DEI D;
    calibrated on the training farm only). No free/driver lr0. At D=240 this is
    exactly 200; D=198 -> 165; D=80 -> 66.667.
  * constant lr for the first ``total_steps//3`` steps, then a
    COMPOUNDING-PRODUCT decay ``lr *= 1/(1+mid*t)`` over the remaining steps,
    with ``mid`` from the SAME bisection pixwake uses so the final lr reaches the
    **ABSOLUTE** ``gamma_min`` (the user-supplied constraint tolerance).
  * ``alpha(t) = alpha0 * D / lr(t)``. Because the skeleton supplies
    ``alpha0 = mean|grad J| / D`` (D-2), this equals ``mean|grad J| / lr(t)`` —
    the native TopFarm coupling — WITHOUT ever forming the driver
    ``mean|grad J| / lr`` alpha0 (which is 1.2x larger at c=0.833).
  * betas 0.1 / 0.2 (TopFarm EasySGDDriver defaults).

At total_steps=6000 (const 2000 / decay 4000) with gamma_min=0.01 this is the
bit-identical twin of ``baseline_schedule.py`` at lr0=c*D — the G1 fidelity gate.
"""
import os
import sys

import jax.numpy as jnp

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pix = os.path.join(_ROOT, "dependencies", "pixwake", "src")
if _pix not in sys.path:
    sys.path.insert(0, _pix)
from pixwake.optim.sgd import _compute_mid_bisection  # noqa: E402

# diameter-rule constant: c = (DEI best lr0) / (DEI D) = 200 / 240
_C = 200.0 / 240.0
_BISECT_LOWER = 0.0
_BISECT_UPPER = 0.1


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    lr0 = _C * float(D)                      # exploration lr scale from D
    n_const = int(total_steps) // 3          # constant lr for first ~1/3
    max_iter = int(total_steps) - n_const    # compounding decay horizon

    # mid depends only on (lr0, gamma_min, max_iter) — all Python floats at
    # trace time, so this is the same precomputed bisection pixwake performs.
    mid = _compute_mid_bisection(
        learning_rate=float(lr0), gamma_min=float(gamma_min),
        max_iter=max_iter, lower=_BISECT_LOWER, upper=_BISECT_UPPER)

    ts = jnp.arange(1, max_iter + 1)                   # 1 .. max_iter
    k = jnp.maximum(step - n_const, 0)                 # decay steps applied
    factors = jnp.where(ts <= k, 1.0 / (1.0 + mid * ts), 1.0)
    lr = lr0 * jnp.prod(factors)                       # compounding product decay

    # native coupling recovered via /D: alpha0 = mean|grad J|/D  ==>
    #   alpha = alpha0 * D / lr = mean|grad J| / lr
    alpha = alpha0 * float(D) / jnp.maximum(lr, 1e-30)
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
