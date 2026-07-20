"""Iter 089: Full cosine, 4x LR, lr/10000, with 5% constant start."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    const_frac = 0.05

    in_const = t < const_frac
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_cos = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))
    lr = jnp.where(in_const, lr_init, lr_cos)

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
