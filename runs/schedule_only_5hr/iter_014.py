"""Iter 014: Larger LR (1.5x) during exploration + aggressive decay.

Hypothesis: larger initial LR helps explore more of the landscape.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    const_frac = 0.4
    in_const = t < const_frac

    # Start at 1.5x lr0 for more exploration
    lr_init = 1.5 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 1499.0 * decay_t)  # decay to lr_init/1500
    lr = jnp.where(in_const, lr_init, lr_decay)

    alpha = jnp.where(in_const,
                      2.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
