"""Iter 054: iter_039 but cosine decay instead of multiplicative.

Same parameters but cosine LR shape to test if decay profile matters.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.35
    in_const = t < const_frac
    lr_init = 3.0 * lr0

    # Cosine decay from lr_init to lr_init/30000
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_min = lr_init / 30000.0
    lr_cos = lr_min + 0.5 * (lr_init - lr_min) * (1 + jnp.cos(jnp.pi * decay_t))
    lr = jnp.where(in_const, lr_init, lr_cos)

    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
