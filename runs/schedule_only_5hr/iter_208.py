"""Iter 208: Transformer-style inverse sqrt schedule.

lr = lr_init * min(t^-0.5, t * warmup^-1.5)
This gives a slower decay than cosine — lr ~1/sqrt(t).
Spends MUCH more time at medium LR, less time at extremes.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_steps = 0.05
    # Inverse sqrt: warmup then 1/sqrt(t)
    warmup_lr = lr_init * t / warmup_steps
    # After warmup: lr = lr_init * sqrt(warmup_steps / t)
    inv_sqrt_lr = lr_init * jnp.sqrt(warmup_steps / jnp.maximum(t, 1e-10))
    inv_sqrt_lr = jnp.maximum(inv_sqrt_lr, lr_min)

    lr_base = jnp.where(t < warmup_steps, warmup_lr, inv_sqrt_lr)

    # Bump at 0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
