"""Iter 236: Two bumps (0.5 and 0.75) + original 5x formula.

Two escape opportunities: first at the midpoint, second later.
Each bump is narrower (sigma=0.04) to minimize disruption.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Two narrow bumps
    bump1 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.50) / 0.04) ** 2)
    bump2 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.04) ** 2)
    lr = lr_base + bump1 + bump2

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
