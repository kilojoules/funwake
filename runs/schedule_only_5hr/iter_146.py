"""Iter 146: Cosine + bump at t=0.7, alpha with sigmoid ramp.

Alpha follows sigmoid instead of 1/lr: stays low longer then ramps sharply.
Lets the optimizer focus on AEP longer before enforcing constraints.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Sigmoid alpha: low early, steep ramp centered at t=0.6
    alpha_max = alpha0 * lr_init / jnp.maximum(lr_min, 1e-10)
    sigmoid = 1.0 / (1.0 + jnp.exp(-15.0 * (t - 0.6)))
    alpha = alpha0 + (alpha_max - alpha0) * sigmoid

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
