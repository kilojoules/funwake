"""Iter 012: Iter_007 + higher beta2=0.9 for better adaptive scaling.

Hypothesis: higher beta2 gives better per-coordinate scaling (standard
Adam behavior) while keeping beta1 low for responsiveness.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Same as iter_007
    const_frac = 0.4
    in_const = t < const_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr0 / (1 + 999.0 * decay_t)
    lr = jnp.where(in_const, lr0, lr_decay)

    alpha = jnp.where(in_const,
                      2.0 * alpha0,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.9  # Higher than iter_007's 0.2

    return lr, alpha, beta1, beta2
