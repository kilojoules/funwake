"""HYPOTHESIS: Replacing the narrow late Gaussian kick with a raised mobility shelf may let turbines continue coordinated wake movement before the final penalty-coupled cooldown.
AXIS: late_mobility_shelf
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Smooth rectangular shelf centered near the known productive late-bump
    # window, but broader and flatter than the previous Gaussian kick.
    rise = 1.0 / (1.0 + jnp.exp(-90.0 * (t - 0.635)))
    fall = 1.0 / (1.0 + jnp.exp(90.0 * (t - 0.755)))
    shelf = rise * fall
    lr = base + 0.22 * lr_init * shelf

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 3.0 * alpha0 * late**2

    return lr, alpha, 0.3, 0.5
