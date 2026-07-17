"""HYPOTHESIS: The near-feasible polynomial schedule needs only a tighter final LR and alpha tail to eliminate the remaining boundary residual.
AXIS: lr_polynomial_decay with stronger alpha_quadratic_ramp.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    hold_frac = 0.22
    min_lr_frac = 0.0015

    phase = jnp.clip((t - hold_frac) / (1.0 - hold_frac), 0.0, 1.0)
    poly = (1.0 - phase) ** 2
    lr_frac = min_lr_frac + (1.0 - min_lr_frac) * poly
    lr = lr0 * jnp.where(t < hold_frac, 1.0, lr_frac)

    alpha_base = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    alpha = alpha_base * (1.0 + 6.0 * t * t)

    return lr, alpha, 0.10, 0.20
