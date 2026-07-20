"""Iter 287: Decoupled alpha (not 1/lr) + proven LR.

Alpha follows its own linear-then-quadratic schedule, independent of lr.
Phase 1 (t<0.5): alpha = alpha0 (constant, low penalty)
Phase 2 (t>=0.5): alpha ramps quadratically from alpha0 to 50*alpha0
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

    # Decoupled alpha: constant early, quadratic ramp late
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha0 * (1.0 + 49.0 * late ** 2)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
