"""Iter 293: Polynomial decay (p=3) with warm restarts at 0.5 and 0.8.

Polynomial decay is slower than cosine initially, keeping LR higher for longer.
Two mini warm restarts give a chance to escape local minima.
Alpha uses square root coupling for gentler constraint enforcement.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 5000.0

    # Polynomial decay base: (1-t)^3
    lr_base = lr_min + (lr_init - lr_min) * (1.0 - t) ** 3

    # Warm restart bumps at t=0.5 and t=0.8
    bump1 = 0.4 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.03) ** 2)
    bump2 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.8) / 0.02) ** 2)
    lr = lr_base + bump1 + bump2

    # Alpha: sqrt coupling (gentler than 1/lr)
    alpha = alpha0 * jnp.sqrt(lr_init / jnp.maximum(lr, 1e-10))
    # Extra quadratic ramp in final 30%
    late = jnp.maximum(t - 0.7, 0.0) / 0.3
    alpha = alpha + 5.0 * alpha0 * late ** 2

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
