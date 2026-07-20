"""Iter 051: Decoupled alpha - polynomial growth independent of LR.

alpha = alpha0 * (1 + k * t^2) instead of 1/lr coupling.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.35
    in_const = t < const_frac
    lr_init = 3.0 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 29999.0 * decay_t)
    lr = jnp.where(in_const, lr_init, lr_decay)

    # Polynomial alpha: starts at 3*alpha0, grows quadratically to ~300*alpha0
    alpha = alpha0 * (3.0 + 297.0 * t ** 2)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
