"""Iter 059: Slower initial decay then faster final decay (S-shaped).

Use sigmoid-like decay to spend more time at intermediate LR.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.35
    in_const = t < const_frac
    lr_init = 3.0 * lr0

    # Sigmoid-like decay: slow start, fast middle, slow end
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    # Map through sigmoid: 0->0, 1->1 but S-shaped
    s = 1.0 / (1.0 + jnp.exp(-12.0 * (decay_t - 0.5)))
    lr_min = lr_init / 30000.0
    lr_sig = lr_init * (1.0 - s) + lr_min * s
    lr = jnp.where(in_const, lr_init, lr_sig)

    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
