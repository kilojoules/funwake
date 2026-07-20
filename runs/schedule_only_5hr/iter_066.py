"""Iter 066: Linear decay with const_frac=0.30."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    const_frac = 0.30
    in_const = t < const_frac
    lr_init = 3.0 * lr0
    lr_min = lr_init / 30000.0

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_lin = lr_init * (1.0 - decay_t) + lr_min * decay_t
    lr = jnp.where(in_const, lr_init, lr_lin)

    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
