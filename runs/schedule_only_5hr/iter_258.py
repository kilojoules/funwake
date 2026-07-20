"""Iter 258: 2-cycle SGDR with gentler peak decay (100%, 40%).

Longer first cycle (60%) for thorough exploration, shorter second (40%)
for convergence. Less aggressive peak decay than iter_256.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cycle 1: [0, 0.6) peak=100%, Cycle 2: [0.6, 1.0) peak=40%
    c1_end = 0.6
    in_c1 = t < c1_end

    t_c1 = t / c1_end
    t_c2 = (t - c1_end) / (1.0 - c1_end)

    lr_c1 = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_c1))
    lr_c2 = lr_min + (0.4 * lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_c2))

    lr = jnp.where(in_c1, lr_c1, lr_c2)

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
