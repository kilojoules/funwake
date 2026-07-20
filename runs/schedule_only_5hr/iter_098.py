"""Iter 098: Full cosine, 4x LR, lr/10000, beta1=0.25, beta2=0.5."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.25
    beta2 = 0.5
    return lr, alpha, beta1, beta2
