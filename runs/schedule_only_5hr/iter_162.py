"""Iter 162: Best recipe with bigger bump (0.5x) and wider (sigma=0.07).

Hypothesis: larger bump = more escape from local minima at t=0.7.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.5 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.07) ** 2)
    lr = lr_base + bump

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
