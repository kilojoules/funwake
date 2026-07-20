"""Iter 058: 3x LR + 3.5x alpha + depth 30k (slightly more alpha)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.35
    in_const = t < const_frac
    lr_init = 3.0 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 29999.0 * decay_t)
    lr = jnp.where(in_const, lr_init, lr_decay)
    alpha = jnp.where(in_const, 3.5 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
