"""Iter 164: Power decay (p=1.5) instead of cosine.

lr = lr_init * (1-t)^1.5 + lr_min — decays slower at start, faster at end.
This is fundamentally different from cosine's symmetric S-curve.
5x alpha coupled, bump at 0.7.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Power decay
    lr_base = lr_min + (lr_init - lr_min) * (1.0 - t) ** 1.5

    # Bump at t=0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
