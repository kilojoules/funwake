"""Iter 186: Alpha dip at t=0.55 with higher base alpha (6x).

Earlier dip when LR is still high, combined with stronger
base alpha to ensure feasibility after the dip.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    alpha_base = 6.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 4.0 * alpha0 * late ** 2

    dip = 0.7 * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
