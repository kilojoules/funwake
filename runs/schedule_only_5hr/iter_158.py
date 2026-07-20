"""Iter 158: Double bump (t=0.5 and t=0.8) + 5x alpha + asymmetric cosine.

Build on iter_153 (best: 5566.1) but add two smaller bumps instead of one,
and use asymmetric cosine (t^0.85) to spend more time at high LR.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Asymmetric cosine — slower initial decay
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t ** 0.85))

    # Two bumps: t=0.5 and t=0.8
    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.8) / 0.04) ** 2)
    lr = lr_base + bump1 + bump2

    # 5x alpha coupled to 1/lr
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
