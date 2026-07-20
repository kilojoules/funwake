"""Iter 180: Flat-top cosine — hold high LR for 25%, then cosine decay.

Maximizes exploration time at full learning rate before the
cosine convergence phase begins. Bump at t=0.75.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Flat for first 25%, then cosine decay
    flat_end = 0.25
    cosine_t = jnp.maximum(t - flat_end, 0.0) / (1.0 - flat_end)
    lr_base = jnp.where(
        t < flat_end,
        lr_init,
        lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    )
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: 5x coupling + quadratic ramp from t=0.5
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
