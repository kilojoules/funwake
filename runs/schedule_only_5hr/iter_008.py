"""Iter 008: Polynomial (quadratic) LR decay + exponential alpha growth.

Polynomial decay gives more control: spends more time at high LR.
Exponential alpha growth ensures very strong feasibility enforcement.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 30% constant, 70% quadratic decay
    const_frac = 0.3
    in_const = t < const_frac

    decay_t = jnp.maximum(t - const_frac, 0.0) / (1.0 - const_frac)
    lr_decay = lr0 * (1.0 - decay_t) ** 2 * (1.0 - 0.99) + lr0 * 0.01
    # Simplify: lr = lr0 * ((1-decay_t)^2 * 0.99 + 0.01)
    lr_decay = lr0 * ((1.0 - decay_t) ** 2 * 0.99 + 0.01)
    lr = jnp.where(in_const, lr0, lr_decay)

    # Exponential alpha growth: alpha0 * exp(5*t) at end = alpha0*e^5 ≈ 148*alpha0
    alpha = alpha0 * jnp.exp(5.0 * t)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
