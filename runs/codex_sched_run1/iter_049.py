"""HYPOTHESIS: The validated late mobility bump may work better with a local Adam
restart during the bump, reducing stale pre-bump momentum while leaving the
successful LR/alpha shape and final polish unchanged.
AXIS: bump_local_adam_restart
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
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 3.0 * alpha0 * late**2

    beta1 = 0.3 - 0.12 * bump_shape
    beta2 = 0.5 - 0.22 * bump_shape

    return lr, alpha, beta1, beta2
