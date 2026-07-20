"""Iter 212: Low alpha early (0.5x), exponential alpha growth.

Radically different alpha: starts at 0.5*alpha0 (heavily relaxed
constraints) and grows exponentially to 20*alpha0. This gives maximum
exploration freedom early and forces strict feasibility late.
LR is proven warmup + cosine + bump.
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

    # Exponential alpha growth: 0.5*alpha0 -> 20*alpha0
    # alpha = 0.5*alpha0 * exp(k*t), k = ln(40) ~ 3.69
    alpha = 0.5 * alpha0 * jnp.exp(3.69 * t)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
