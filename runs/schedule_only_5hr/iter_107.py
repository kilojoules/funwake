"""Iter 107: Cosine LR, 4x, lr/10000, lower initial alpha (0.5*alpha0)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    # Start with half alpha, still scale with 1/lr
    alpha = 0.5 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
