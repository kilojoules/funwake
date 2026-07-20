"""Iter 018: 1.5x LR + 2x alpha + deep decay (iter_014 fix attempt).

Iter_014 was 5533.46 but infeasible. This adds stronger alpha and deeper decay.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    const_frac = 0.35
    in_const = t < const_frac

    lr_init = 1.5 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    # Decay to lr_init/3000
    lr_decay = lr_init / (1 + 2999.0 * decay_t)
    lr = jnp.where(in_const, lr_init, lr_decay)

    # Moderate alpha during const, 1/lr coupling (reaches ~3000*alpha0)
    alpha = jnp.where(in_const,
                      2.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
