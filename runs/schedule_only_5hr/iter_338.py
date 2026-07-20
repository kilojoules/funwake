"""Iter 338: Wider 30% plateau + shifted bumps later + stronger feas.

Hypothesis: longer exploration at full LR, then faster decay. Bumps
shifted slightly later. Stronger final feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.30
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.58) / 0.04) ** 2)
    bump2 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.80) / 0.03) ** 2)
    feas1 = 0.12 * lr_init * jnp.exp(-0.5 * ((t - 0.86) / 0.02) ** 2)
    feas2 = 0.08 * lr_init * jnp.exp(-0.5 * ((t - 0.93) / 0.015) ** 2)
    lr = lr_base + bump1 + bump2 + feas1 + feas2

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.5 * alpha0 * late ** 2
    fb1 = 15.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.86) / 0.02) ** 2)
    fb2 = 28.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.93) / 0.015) ** 2)
    alpha = alpha_base + alpha_extra + fb1 + fb2

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
