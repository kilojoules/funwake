"""Iter 016: 2x LR exploration + 3x alpha enforcement + very deep decay.

Building on iter_014's success: even larger LR but with much stronger
constraint enforcement to achieve feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    const_frac = 0.35
    in_const = t < const_frac

    lr_init = 2.0 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    # Very deep decay: 2*lr0 -> 2*lr0/5000
    lr_decay = lr_init / (1 + 4999.0 * decay_t)
    lr = jnp.where(in_const, lr_init, lr_decay)

    # Strong alpha: 3x during exploration, then 1/lr coupling
    alpha = jnp.where(in_const,
                      3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
