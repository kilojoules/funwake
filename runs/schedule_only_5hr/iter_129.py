"""Iter 129: Cosine LR, 4x, lr/10000, beta1=0.3, beta2=0.5, alpha = alpha0 * (1 + (lr_init/lr - 1)*0.5).

Dampened 1/lr coupling: only half the coupling strength.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    ratio = lr_init / jnp.maximum(lr, 1e-10)
    # Dampened coupling: alpha increases at half the rate
    alpha = alpha0 * (1.0 + (ratio - 1.0) * 0.5)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
