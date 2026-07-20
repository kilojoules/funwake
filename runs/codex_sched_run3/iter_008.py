"""HYPOTHESIS: A middle final polish boost can pass held-out boundary feasibility without overshooting the stressed quick polygon.
AXIS: lr_polynomial_decay with balanced final-window alpha boost.
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
    polish = jnp.clip((t - 0.90) / 0.10, 0.0, 1.0)
    alpha = alpha_base * (1.0 + 10.0 * t * t + 80.0 * polish * polish)

    return lr, alpha, 0.10, 0.20
