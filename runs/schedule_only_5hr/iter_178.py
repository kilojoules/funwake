"""Iter 178: 5x LR + exponential alpha ramp + decreasing beta1.

Higher initial LR for more exploration, exponential alpha for
sharper late-stage constraint enforcement, beta1 smoothly
decreasing from 0.4 to 0.1 for fine-grained late convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0

    # Cosine LR decay with bump at 0.7
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Exponential alpha ramp: starts at 3x, grows exponentially to ~15x
    alpha_mult = 3.0 * jnp.exp(1.6 * t)  # 3.0 -> ~14.8
    alpha = alpha_mult * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Smoothly decreasing beta1
    beta1 = 0.4 - 0.3 * t  # 0.4 -> 0.1
    beta2 = 0.5

    return lr, alpha, beta1, beta2
