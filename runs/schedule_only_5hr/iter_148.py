"""Iter 148: Cosine LR with super-convergence style momentum schedule.

beta1: 0.5 -> 0.1 -> 0.5 (high-low-high). High momentum at start for big moves,
low in middle for precise refinement, high at end for stable convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # V-shaped beta1: starts high, dips at t=0.5, back high
    beta1 = 0.5 - 0.4 * jnp.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
    # This gives: ~0.5 at t=0, ~0.1 at t=0.5, ~0.5 at t=1
    beta2 = 0.5

    return lr, alpha, beta1, beta2
