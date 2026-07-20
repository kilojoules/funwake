"""Iter 010: Long exploration (50%) + very deep decay + aggressive alpha.

Build on iter_007's success: longer constant phase, deeper final decay.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 50% constant, 50% decay
    const_frac = 0.5
    in_const = t < const_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    # Very deep decay: lr0 -> lr0/2000
    lr_decay = lr0 / (1 + 1999.0 * decay_t)
    lr = jnp.where(in_const, lr0, lr_decay)

    # Alpha: moderate during explore, very aggressive during decay
    alpha = jnp.where(in_const,
                      1.5 * alpha0,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
