"""Iter 223: Logarithmic alpha growth instead of coupled.

Previous attempts couple alpha to 1/lr. Try a fundamentally different
alpha schedule: logarithmic growth from alpha0 to 30*alpha0. This
decouples alpha from LR, giving the optimizer more freedom early.
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

    # Logarithmic alpha: decoupled from LR
    # alpha = alpha0 * (1 + 29 * log(1 + 9*t) / log(10))
    alpha = alpha0 * (1.0 + 29.0 * jnp.log(1.0 + 9.0 * t) / jnp.log(10.0))

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
