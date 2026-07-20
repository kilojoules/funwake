"""Iter 090: Asymmetric cosine (power 0.5) — stays at high LR longer. 4x, lr/10000."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Power-cosine: compress early decay, steepen late decay
    # cos(pi * t^0.5) makes LR stay high longer
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t ** 0.5))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
