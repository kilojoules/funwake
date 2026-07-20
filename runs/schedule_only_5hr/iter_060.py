"""Iter 060: Phase-transition beta: low beta1 during explore, higher during decay.

Hypothesis: higher momentum during decay helps with constraint enforcement.
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
    alpha = jnp.where(in_const, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    # Phase-transition betas
    beta1 = jnp.where(in_const, 0.1, 0.3)
    beta2 = jnp.where(in_const, 0.2, 0.4)
    return lr, alpha, beta1, beta2
