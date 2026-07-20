"""Iter 118: Cosine LR with power 1.5 (mild asymmetry). 4x, lr/10000, (0.3,0.5)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Power 1.5: slightly asymmetric, stays higher early
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t ** 1.5))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
