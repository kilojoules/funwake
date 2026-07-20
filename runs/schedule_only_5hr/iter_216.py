"""Iter 216: Alpha proportional to step (linear growth) instead of coupled.

Instead of alpha ~ 1/lr (which can spike when lr is near zero),
use a simple linear growth: alpha = alpha0 * (1 + k*t).
This is a totally different alpha philosophy — smooth, predictable growth.
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

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Linear alpha growth: from alpha0 to 50*alpha0
    alpha = alpha0 * (1.0 + 49.0 * t)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
