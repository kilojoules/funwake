"""Iter 194: Extended flat-top (5-35%) + cosine + bump at 0.8.

Long constant-LR phase for extended exploration. Bump later (0.8)
because the cosine decay starts later (at 35% instead of 5%).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    flat_end = 0.35
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - flat_end) / (1.0 - flat_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr,
              jnp.where(t < flat_end, lr_init, cosine_lr))

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.8) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
