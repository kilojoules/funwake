"""Iter 278: Proven formula but higher lr_min (lr_init/1000 vs /10000).

Higher minimum LR means the optimizer can still make small adjustments
at the end, instead of effectively freezing. Less constraint enforcement
but potentially better fine-tuning.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 1000.0  # 10x higher than usual

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Slightly stronger alpha to compensate for higher lr_min
    alpha_base = 6.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 4.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
