"""Iter 296: Linear warmup (10%) + exponential decay with bump + heavy late alpha.

Different from cosine: exponential decay from high LR gives different
exploration-exploitation tradeoff. Linear warmup is longer for better init.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_final = 0.005 * lr0

    # 10% linear warmup
    warmup_end = 0.1
    warmup_lr = lr_init * t / warmup_end

    # Exponential decay from warmup_end to 1.0
    decay_t = jnp.clip((t - warmup_end) / (1.0 - warmup_end), 0.0, 1.0)
    exp_lr = lr_init * jnp.exp(decay_t * jnp.log(lr_final / lr_init))

    lr_base = jnp.where(t < warmup_end, warmup_lr, exp_lr)

    # Bump at 0.65
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.65) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: inverse coupling + quadratic late ramp
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 5.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
