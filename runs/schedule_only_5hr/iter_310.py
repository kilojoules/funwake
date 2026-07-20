"""Iter 310: Plateau + bumps with alpha dips during bumps.

When LR bumps up, temporarily reduce alpha to allow larger moves,
then alpha recovers. This gives true "mini restarts" where
constraints relax briefly then re-tighten.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.25
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    bump2 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.78) / 0.03) ** 2)
    lr = lr_base + bump1 + bump2

    # Alpha: inverse coupling + late ramp, but dip during bumps
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    # Alpha dips (negative Gaussians) at bump positions
    dip1 = 0.5 * jnp.exp(-0.5 * ((t - 0.55) / 0.03) ** 2)
    dip2 = 0.3 * jnp.exp(-0.5 * ((t - 0.78) / 0.02) ** 2)
    alpha_mult = 1.0 - dip1 - dip2

    alpha = jnp.maximum(alpha_base * alpha_mult + alpha_extra, alpha0 * 0.5)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
