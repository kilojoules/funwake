"""Dual-bump schedule with later final settling.

HYPOTHESIS: Iteration 15 proves the high-LR dual-bump strategy works; a later
settling gate may preserve more of the second bump's AEP gain while still
leaving enough constraint-focused steps for feasibility.
AXIS: lr_gaussian_bumps and alpha_anti_phase_dip with a later final LR taper
than iter_015.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_init = 4.0 * lr0
    warm_len = 0.05
    warm_t = jnp.minimum(1.0, t / warm_len)
    warm_t = warm_t * warm_t * (3.0 - 2.0 * warm_t)

    decay_t = jnp.minimum(1.0, jnp.maximum(0.0, (t - warm_len) / (1.0 - warm_len)))
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))
    floor = 0.002
    lr_cos = lr_init * (floor + (1.0 - floor) * cosine)
    lr_warm = lr_init * (0.15 + 0.85 * warm_t)
    lr_base = jnp.where(t < warm_len, lr_warm, lr_cos)

    bump1 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.50) / 0.040) ** 2)
    bump2 = 0.30 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.050) ** 2)

    settle = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.80) / 0.20))
    settle = settle * settle * (3.0 - 2.0 * settle)
    lr = (lr_base + bump1 + bump2) * (1.0 - 0.96 * settle)

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.70) / 0.30))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 8.0 * late + 60.0 * late * late)

    dip = jnp.exp(-0.5 * ((t - 0.60) / 0.070) ** 2)
    alpha = alpha * (1.0 - 0.50 * dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
