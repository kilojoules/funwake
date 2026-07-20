"""Iter 109: Cosine LR, 4x, lr/10000, betas increase from (0.1,0.2) to (0.5,0.8)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Increase betas over time: low momentum early, high late
    beta1 = 0.1 * (1.0 - t) + 0.5 * t
    beta2 = 0.2 * (1.0 - t) + 0.8 * t
    return lr, alpha, beta1, beta2
