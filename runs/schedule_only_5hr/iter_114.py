"""Iter 114: Cosine LR, 4x, lr/10000, sigmoid alpha ramp (decoupled from LR)."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Sigmoid alpha: smoothly ramps from alpha0 to 100*alpha0
    # Centered at t=0.5 with steepness 10
    sigmoid = 1.0 / (1.0 + jnp.exp(-10.0 * (t - 0.5)))
    alpha = alpha0 * (1.0 + 99.0 * sigmoid)

    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
