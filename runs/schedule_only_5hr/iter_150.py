"""Iter 150: Dual cosine (two full cycles) with decaying amplitude.

Two full cosine cycles: cycle 1 steps 0-4000, cycle 2 steps 4000-8000.
Second cycle has 0.5x the LR range. Like SGDR but with equal-length cycles.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Two equal-length cosine cycles
    in_cycle1 = t < 0.5
    local_t1 = t / 0.5
    local_t2 = (t - 0.5) / 0.5

    lr_c1 = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * local_t1))
    lr_c2_max = 0.5 * lr_init
    lr_c2 = lr_min + (lr_c2_max - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * local_t2))

    lr = jnp.where(in_cycle1, lr_c1, lr_c2)

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
