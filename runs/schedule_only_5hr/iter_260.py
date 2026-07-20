"""Iter 260: Exponential decay with periodic LR pulses.

Base: exponential decay lr_init * exp(-5t).
Pulses: sharp Gaussian spikes at t=0.3, 0.5, 0.7 to escape local minima.
Different from cosine — decays faster early, slower late.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Exponential decay
    lr_base = lr_min + (lr_init - lr_min) * jnp.exp(-5.0 * t)

    # Three escape pulses
    pulse1 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.3) / 0.03) ** 2)
    pulse2 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    pulse3 = 0.10 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.03) ** 2)
    lr = lr_base + pulse1 + pulse2 + pulse3

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
