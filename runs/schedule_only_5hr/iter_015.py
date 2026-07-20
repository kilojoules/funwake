"""Iter 015: Step-wise decay (3 plateaus) + aggressive alpha.

Instead of smooth decay, use 3 LR plateaus that drop sharply.
This gives more optimization time at each LR level.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 3 plateaus: 0-40% at lr0, 40-70% at 0.1*lr0, 70-100% at 0.01*lr0
    p1 = t < 0.4
    p2 = (t >= 0.4) & (t < 0.7)

    lr = jnp.where(p1, lr0,
         jnp.where(p2, 0.1 * lr0,
                   0.01 * lr0))

    # Alpha: inverse coupled to LR
    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
