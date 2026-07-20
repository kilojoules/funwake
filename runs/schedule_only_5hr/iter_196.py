"""Iter 196: Quadratic LR decay (not cosine) + warmup + bump.

Fundamentally different from cosine: lr = lr_init * (1-t')^2.
Quadratic starts slow, accelerates late — more time near peak LR.
Same proven alpha and beta structure.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup (0-5%), then quadratic decay
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    decay_t = (t - warmup_end) / (1.0 - warmup_end)
    quad_lr = lr_min + (lr_init - lr_min) * (1.0 - decay_t) ** 2

    lr_base = jnp.where(t < warmup_end, warmup_lr, quad_lr)

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
