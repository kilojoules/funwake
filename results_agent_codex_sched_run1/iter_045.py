"""HYPOTHESIS: An asymmetric late LR kick with a sharper onset and longer trailing tail may preserve the best basin while giving more post-kick wake polishing than a symmetric Gaussian.
AXIS: asymmetric_late_bump_tail
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    center = 0.692
    left_w = 0.038
    right_w = 0.060
    width = jnp.where(t < center, left_w, right_w)
    bump = 0.285 * lr_init * jnp.exp(-0.5 * ((t - center) / width) ** 2)
    lr = lr_base + bump

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 3.0 * alpha0 * late**2

    return lr, alpha, 0.3, 0.5
