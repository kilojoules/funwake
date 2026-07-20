"""Iter 119: Cosine alpha (not 1/lr) + cosine LR. Both cosine. 4x, lr/10000, (0.3,0.5)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Alpha also follows cosine but INCREASES (low early, high late)
    alpha_max = alpha0 * 100.0
    alpha = alpha0 + (alpha_max - alpha0) * 0.5 * (1.0 - jnp.cos(jnp.pi * t))

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
