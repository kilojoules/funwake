"""HYPOTHESIS: After the proven main mobility bump settles the basin, a small
late reheat with its own constraint pulse may allow final wake polishing without
loosening feasibility for the rest of the run.
AXIS: post_bump_late_polish_reheat
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    main_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    polish_bump = jnp.exp(-0.5 * ((t - 0.835) / 0.022) ** 2)
    lr = lr_base + 0.3 * lr_init * main_bump + 0.035 * lr_init * polish_bump

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 1.18 * alpha0 * main_bump
    alpha = alpha + 0.75 * alpha0 * polish_bump

    return lr, alpha, 0.3, 0.5
