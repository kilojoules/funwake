"""HYPOTHESIS: A slightly wider local alpha counter-ramp may smooth the same
pressure increase without the broad-shape penalty seen at width 0.065.
AXIS: bump_constraint_counter_ramp_width_0p055
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    alpha_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.055) ** 2)
    lr = lr_base + 0.3 * lr_init * lr_bump
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2
    alpha = alpha + 1.2 * alpha0 * alpha_bump
    return lr, alpha, 0.3, 0.5
