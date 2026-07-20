"""Iter 052: Exponential alpha growth + iter_039 LR.

alpha = alpha0 * exp(k*t) for smooth exponential growth.
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

    # Exponential alpha: 3*alpha0 at t=0, ~3*alpha0*e^8 ≈ 8000*alpha0 at t=1
    alpha = 3.0 * alpha0 * jnp.exp(8.0 * t)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
