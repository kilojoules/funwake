"""Iter 222: Smooth beta transition + proven cosine.

Instead of constant betas, smoothly transition beta1 from 0.6 -> 0.1
and beta2 from 0.8 -> 0.3. Higher momentum early for exploration,
near-SGD late for precision. Same 4x LR + cosine + bump backbone.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    # Smooth beta transition
    beta1 = 0.6 - 0.5 * t   # 0.6 -> 0.1
    beta2 = 0.8 - 0.5 * t   # 0.8 -> 0.3

    return lr, alpha, beta1, beta2
