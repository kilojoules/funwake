"""Iter 175: Standard Adam (beta1=0.9, beta2=0.999) with cosine decay.

All prior attempts used low beta values (0.1-0.5). Standard Adam
has much higher momentum and adaptive scaling. This could better
traverse flat regions and find better optima.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Simple cosine decay (no bump)
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Coupled alpha
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Standard Adam parameters
    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
