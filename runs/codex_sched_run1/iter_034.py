"""HYPOTHESIS: The productive late LR bump may improve if alpha only partially relaxes during the bump, preserving more constraint pressure while still allowing wake-driven movement.
AXIS: partial_alpha_decoupled_bump
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_lr = lr_base + 0.35 * bump
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(alpha_lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2

    return lr, alpha, 0.3, 0.5
