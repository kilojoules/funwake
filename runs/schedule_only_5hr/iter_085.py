"""Iter 085: Full cosine, 4x LR, lr/10000, try alpha with quadratic ramp."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Quadratic alpha ramp: starts at alpha0, ramps up quadratically
    alpha = alpha0 * (1.0 + 50.0 * t ** 2)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
