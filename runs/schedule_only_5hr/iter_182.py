"""Iter 182: LR bump + alpha dip at t=0.7 (combined re-exploration).

Both increase effective step size AND relax constraints
simultaneously for maximum re-exploration before final convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cosine LR with bump at 0.7
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha with coupling + quadratic + dip at 0.7
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    dip = 0.5 * jnp.exp(-0.5 * ((t - 0.7) / 0.04) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
