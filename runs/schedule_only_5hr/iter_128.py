"""Iter 128: Cosine LR with standard Adam (0.9, 0.999). 4x, lr/10000."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.9
    beta2 = 0.999
    return lr, alpha, beta1, beta2
