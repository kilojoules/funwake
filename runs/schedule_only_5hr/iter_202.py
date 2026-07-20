"""Iter 202: Flat-top cosine — hold peak LR for 20% before decaying.

Instead of immediately decaying after warmup, hold the peak LR for a
significant portion (5% warmup + 20% flat + 75% cosine decay).
More time at peak = more exploration before settling.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    flat_end = 0.25  # hold peak until 25%

    warmup_lr = lr_init * t / warmup_end
    decay_t = (t - flat_end) / (1.0 - flat_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))

    lr_base = jnp.where(t < warmup_end, warmup_lr,
              jnp.where(t < flat_end, lr_init, cosine_lr))

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
