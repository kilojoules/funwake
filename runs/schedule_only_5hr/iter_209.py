"""Iter 209: Cubic polynomial decay (slower than cosine at start).

lr = lr_init * (1 - t')^3  after warmup.
Cubic stays near peak much longer than cosine or quadratic,
then drops very rapidly at the end. More exploration budget.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    decay_t = (t - warmup_end) / (1.0 - warmup_end)
    cubic_lr = lr_min + (lr_init - lr_min) * (1.0 - decay_t) ** 3

    lr_base = jnp.where(t < warmup_end, warmup_lr, cubic_lr)

    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
