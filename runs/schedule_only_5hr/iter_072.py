"""Iter 072: Full cosine decay with 3.5x LR (no constant phase)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 3.5 * lr0
    lr_min = lr_init / 30000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
