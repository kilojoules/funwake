"""Iter 213: Cosine + periodic mini-bumps every 15%.

Instead of one big bump at 0.7, add small periodic bumps at
0.3, 0.45, 0.6, 0.75, 0.9 — cyclical perturbations to escape
local minima repeatedly throughout optimization.
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

    # Periodic mini-bumps (decreasing amplitude)
    bump = 0.0
    for center, amp in [(0.3, 0.15), (0.45, 0.12), (0.6, 0.10), (0.75, 0.08), (0.9, 0.05)]:
        bump = bump + amp * lr_init * jnp.exp(-0.5 * ((t - center) / 0.03) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
