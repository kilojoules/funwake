"""Iter 117: Cosine LR 4x lr/10000, alpha = alpha0 * (lr_init/lr)^1.5 (stronger coupling)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    # Stronger coupling
    ratio = lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha0 * ratio ** 1.5

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
