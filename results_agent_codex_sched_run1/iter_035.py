"""HYPOTHESIS: A small pre-bump plus localized alpha dip can reshape the layout before the proven late bump, testing a two-kick path instead of another late-center refinement.
AXIS: dual_bump_local_alpha_dip
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    pre_bump = 0.10 * lr_init * jnp.exp(-0.5 * ((t - 0.49) / 0.035) ** 2)
    late_bump = 0.30 * lr_init * jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    lr = lr_base + pre_bump + late_bump

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2
    dip = 0.12 * jnp.exp(-0.5 * ((t - 0.58) / 0.055) ** 2)
    alpha = alpha * (1.0 - dip)

    return lr, alpha, 0.3, 0.5
