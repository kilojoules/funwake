"""Iter 126: Cosine LR, linear alpha ramp (decoupled). 4x, lr/10000, (0.3,0.5)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Linear alpha ramp from alpha0 to 50*alpha0
    alpha = alpha0 * (1.0 + 49.0 * t)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
