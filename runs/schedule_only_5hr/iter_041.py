"""Iter 041: 3x LR + ramping alpha (2x->4x) during explore + depth 20000."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    const_frac = 0.35
    in_const = t < const_frac

    lr_init = 3.0 * lr0
    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr_init / (1 + 19999.0 * decay_t)
    lr = jnp.where(in_const, lr_init, lr_decay)

    # Alpha ramps during exploration from 2x to 4x
    explore_t = t / jnp.maximum(const_frac, 1e-6)
    alpha_explore = alpha0 * (2.0 + 2.0 * explore_t)

    alpha = jnp.where(in_const,
                      alpha_explore,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
