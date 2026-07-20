"""Iter 007: Very aggressive alpha + longer constant + low betas.

The seed itself is infeasible. Key fix: much higher alpha in late phase.
Start with moderate alpha, ramp to 500x.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 40% constant, 60% multiplicative decay to 0.001*lr0
    const_frac = 0.4
    in_const = t < const_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    # Multiplicative decay: lr0 / (1 + 999*decay_t) -> final = lr0/1000
    lr_decay = lr0 / (1 + 999.0 * decay_t)
    lr = jnp.where(in_const, lr0, lr_decay)

    # Alpha: start at 2*alpha0, end at 500*alpha0
    alpha = jnp.where(in_const,
                      2.0 * alpha0,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
