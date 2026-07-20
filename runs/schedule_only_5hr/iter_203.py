"""Iter 203: Exponential decay with very high initial LR (6x).

Exponential has a different decay profile than cosine — it spends
proportionally MORE time at intermediate LR values.
lr = lr_init * exp(-k*t) where k is tuned for 99% reduction.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 6.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + exponential decay
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end

    # k such that exp(-k * 0.95) = 0.01 => k = -ln(0.01)/0.95 ~ 4.85
    k = 4.85
    decay_t = (t - warmup_end) / (1.0 - warmup_end)
    exp_lr = lr_init * jnp.exp(-k * decay_t)
    exp_lr = jnp.maximum(exp_lr, lr_min)

    lr_base = jnp.where(t < warmup_end, warmup_lr, exp_lr)

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
