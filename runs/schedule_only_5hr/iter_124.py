"""Iter 124: Cosine LR + two small bumps at t=0.4 and t=0.7. 4x, lr/10000, (0.3,0.5)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.4) / 0.04) ** 2)
    bump2 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.04) ** 2)
    lr = lr_base + bump1 + bump2

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
