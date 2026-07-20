"""Iter 147: Cosine LR + alpha dip at t=0.5 then steep ramp.

Alpha briefly drops at t=0.5 to let optimizer escape boundary traps,
then ramps back up steeply for final convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: 1/lr coupling but with a dip at t=0.5
    base_alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    dip = 0.5 * jnp.exp(-0.5 * ((t - 0.5) / 0.08) ** 2)
    alpha = base_alpha * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
