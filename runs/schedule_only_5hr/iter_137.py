"""Iter 137: Cosine + bump at t=0.7, 4x lr0, with alpha quadratic coupling.

Instead of alpha proportional to 1/lr, use sqrt(1/lr) for slower alpha ramp.
This keeps constraint pressure moderate longer, giving more room to optimize AEP.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Sqrt coupling: alpha grows slower than 1/lr
    ratio = lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha0 * jnp.sqrt(ratio)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
