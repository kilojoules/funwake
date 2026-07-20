"""Iter 286: Exponential alpha ramp in final 15% + proven LR formula.

Instead of quadratic tail, use exponential ramp for aggressive
feasibility enforcement at the very end.
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

    # Base coupling
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    # Exponential ramp in final 15%: goes from 1x to ~20x
    final_ramp = jnp.maximum(t - 0.85, 0.0) / 0.15
    alpha_exp = alpha0 * (jnp.exp(3.0 * final_ramp) - 1.0)
    # Moderate quadratic from t=0.5
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_quad = 2.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_quad + alpha_exp

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
