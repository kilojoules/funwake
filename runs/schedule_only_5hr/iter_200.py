"""Iter 200: Higher initial LR (5x) + double bump at 0.5 and 0.75.

Push harder early, with two recovery bumps instead of one.
The first bump at 0.5 gives a mid-optimization exploration kick,
the second at 0.75 prevents premature convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Double bump
    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.04) ** 2)
    lr = lr_base + bump1 + bump2

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
