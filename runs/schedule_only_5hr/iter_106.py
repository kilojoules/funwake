"""Iter 106: Cosine LR, 4x, lr/10000, betas decay from (0.5,0.7) to (0.1,0.2)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Linearly decay betas from high to low
    beta1 = 0.5 * (1.0 - t) + 0.1 * t
    beta2 = 0.7 * (1.0 - t) + 0.2 * t
    return lr, alpha, beta1, beta2
