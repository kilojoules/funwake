"""Iter 013: Cyclic alpha exploration (first 50%) + aggressive enforcement (last 50%).

Combine iter_011's cyclic alpha with iter_007's aggressive 1/lr coupling.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # LR: constant for 40%, then deep multiplicative decay
    const_frac = 0.4
    in_const = t < const_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr0 / (1 + 999.0 * decay_t)
    lr = jnp.where(in_const, lr0, lr_decay)

    # Alpha: cyclic during constant phase, then 1/lr coupling
    # Cyclic: 2 oscillations during constant phase
    cycle_t = t / jnp.maximum(const_frac, 1e-6)
    cycle = jnp.sin(2 * jnp.pi * 2 * cycle_t)
    alpha_explore = alpha0 * (1.5 + 1.0 * cycle)  # range: 0.5 to 2.5 * alpha0

    alpha_enforce = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    alpha = jnp.where(in_const, alpha_explore, alpha_enforce)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
