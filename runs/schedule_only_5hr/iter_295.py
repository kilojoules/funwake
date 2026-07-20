"""Iter 295: Asymmetric cosine (t^0.7) + exponential alpha ramp in final 20%.

Based on iter_262 (5599.97 infeasible). Fix: massive alpha enforcement
in the last 20% using exponential ramp to force feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    # Asymmetric: t^0.7 keeps LR higher for longer
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t ** 0.7))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: inverse coupling + exponential ramp in final 20%
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.8, 0.0) / 0.2
    # Exponential ramp: goes from 0 to 50*alpha0 in last 20%
    alpha_extra = 50.0 * alpha0 * (jnp.exp(3.0 * late) - 1.0) / (jnp.exp(3.0) - 1.0)
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
