"""HYPOTHESIS: A 1.25x local alpha counter-ramp may improve the previous best
if the useful pressure peak is slightly above 1.2x.
AXIS: bump_constraint_counter_ramp_strength_1p25
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
    alpha = alpha + 1.25 * alpha0 * bump_shape
    return lr, alpha, 0.3, 0.5
