"""Iter 299: Polynomial decay (t^2) + stronger alpha coupling.

Polynomial (1-t)^2 decays slower than cosine initially → more exploration.
Alpha uses (lr_init/lr)^1.3 for slightly stronger constraint at low LR.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end

    # Polynomial decay: (1-t')^2
    poly_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    poly_lr = lr_min + (lr_init - lr_min) * (1.0 - poly_t) ** 2

    lr_base = jnp.where(t < warmup_end, warmup_lr, poly_lr)

    # Bump at 0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: power coupling + late ramp
    ratio = lr_init / jnp.maximum(lr, 1e-10)
    alpha_base = 5.0 * alpha0 * ratio ** 1.3
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 5.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
