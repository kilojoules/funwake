"""Iter 135: Cosine + large bump at t=0.65, 4.5x lr0.

Slightly higher initial LR (4.5x vs 4x), larger bump (0.5x) at t=0.65.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.5 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.5 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    lr = lr_base + bump

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
