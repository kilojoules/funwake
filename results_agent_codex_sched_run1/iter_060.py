"""HYPOTHESIS: A 1.1x local alpha counter-ramp may sit closer to the narrow
optimum than the 1.2x version while keeping the validated LR path unchanged.
AXIS: bump_constraint_counter_ramp_strength_1p1
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump_shape = jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    lr = lr_base + 0.3 * lr_init * bump_shape
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2
    alpha = alpha + 1.1 * alpha0 * bump_shape
    return lr, alpha, 0.3, 0.5
