"""Iter 206: Sigmoid alpha transition instead of quadratic.

Replace the quadratic late ramp with a sharp sigmoid at t=0.7.
This creates a more abrupt transition from exploration to feasibility
enforcement, leaving more optimization budget for pure AEP maximization.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine (proven base)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Bump at 0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: base coupling + sigmoid transition at t=0.7
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    sigmoid = 1.0 / (1.0 + jnp.exp(-25.0 * (t - 0.7)))
    alpha_extra = 8.0 * alpha0 * sigmoid
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
