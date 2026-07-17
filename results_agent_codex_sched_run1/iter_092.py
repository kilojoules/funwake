"""HYPOTHESIS: A plateau-cosine schedule with explicit late feasibility bursts
is a different strategy from the transferred dual-bump incumbent. It keeps
early mobility high, uses two narrower wake-seeking bumps, then applies
localized alpha bursts near the end to repair spacing and boundary drift.
AXIS: plateau_double_bump_feasibility_bursts
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.22
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    bump2 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.78) / 0.03) ** 2)
    feas1 = 0.12 * lr_init * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    feas2 = 0.08 * lr_init * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)
    lr = lr_base + bump1 + bump2 + feas1 + feas2

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late**2
    burst1 = 15.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    burst2 = 25.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)
    alpha = alpha_base + alpha_extra + burst1 + burst2

    return lr, alpha, 0.3, 0.5
