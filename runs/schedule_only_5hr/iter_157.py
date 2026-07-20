"""Iter 157: Exponential decay LR + 7x alpha + bump at 0.65.

Instead of cosine, use exponential decay: lr = lr_init * exp(-6*t).
This decays faster early and has a longer tail at low LR.
Higher alpha (7x) and bump slightly earlier (0.65).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Exponential decay
    lr_base = lr_min + (lr_init - lr_min) * jnp.exp(-6.0 * t)

    # Bump at t=0.65
    bump = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    lr = lr_base + bump

    # 7x alpha coupled to 1/lr
    alpha = 7.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
