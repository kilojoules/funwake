"""Iter 163: Quadratic alpha ramp (not just 1/lr).

Alpha = base (coupled to 1/lr) + quadratic time component.
This makes alpha grow faster than 1/lr in the last 30%, ensuring
very strict feasibility at convergence.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Base alpha coupled to 1/lr
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Extra quadratic ramp for last 30%
    late = jnp.maximum(t - 0.7, 0.0) / 0.3  # 0 to 1 in last 30%
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
