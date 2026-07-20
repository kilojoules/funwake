"""Iter 143: Cosine + bump at t=0.65 with beta1 drop + wider bump.

Combining best ideas: bump with temporary beta1 reduction, wider Gaussian.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump_weight = jnp.exp(-0.5 * ((t - 0.65) / 0.07) ** 2)
    bump = 0.4 * lr_init * bump_weight
    lr = lr_base + bump

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3 - 0.2 * bump_weight
    beta2 = 0.5

    return lr, alpha, beta1, beta2
