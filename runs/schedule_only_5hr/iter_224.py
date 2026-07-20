"""Iter 224: 5x LR + dual bumps + slightly lower alpha for more AEP.

Iter_221 got 5565.43 feasible with 6x alpha coupling. Try reducing to
5.5x to give more AEP headroom while keeping feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.8) / 0.04) ** 2)
    lr = lr_base + bump1 + bump2

    # Slightly lower alpha base
    alpha_base = 5.5 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.5 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
