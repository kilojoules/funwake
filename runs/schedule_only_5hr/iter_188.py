"""Iter 188: Flat-top cosine + simultaneous bump+dip at t=0.65.

Flat-top: warmup (0-5%), constant max LR (5-20%), cosine decay (20-100%).
Keeps exploration at full power 3x longer than warmup-only.
Simultaneous LR bump and alpha dip at t=0.65 for coordinated re-exploration.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup (0-5%), flat top (5-20%), cosine decay (20-100%)
    warmup_end = 0.05
    flat_end = 0.20
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - flat_end) / (1.0 - flat_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))

    lr_base = jnp.where(t < warmup_end, warmup_lr,
              jnp.where(t < flat_end, lr_init, cosine_lr))

    # LR bump at t=0.65
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: 5x coupling + quadratic ramp + dip at t=0.65
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    dip = 0.6 * jnp.exp(-0.5 * ((t - 0.65) / 0.04) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
