"""HYPOTHESIS: Moving the improved earlier bump a little farther forward may give even more time for late penalty-coupled polishing.
AXIS: cosine_bump_center_refine
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr = lr_base + 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.675) / 0.05) ** 2)
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2
    return lr, alpha, 0.3, 0.5
