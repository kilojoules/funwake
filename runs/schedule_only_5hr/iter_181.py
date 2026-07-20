"""Iter 181: Alpha dip at t=0.7 instead of LR bump.

Briefly relax constraints (reduce alpha by 70%) around t=0.7
to let turbines re-explore positions, then snap back for
final convergence. No LR bump.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Pure cosine LR decay (no bump)
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Alpha with coupling + quadratic ramp + dip at t=0.7
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    # Dip: reduce alpha by 70% in a narrow window around t=0.7
    dip = 0.7 * jnp.exp(-0.5 * ((t - 0.7) / 0.03) ** 2)
    alpha = (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
