"""HYPOTHESIS: The rectangular shelf was feasible but lost too much AEP. Return
to the distinct strategy that consistently beats the last scored single-bump
attempt: a warmup cosine backbone with two broad mobility windows, inverse-LR
penalty coupling, and a strong late quadratic feasibility recovery.
AXIS: restored_validated_dual_window_after_shelf_miss
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (t - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    mid_shape = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_shape = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr = lr_base + 0.458205 * lr_init * mid_shape + 0.165784 * lr_init * late_shape
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 2.879478 * alpha0 * lr_init / lr
    alpha = alpha + 16.850946 * alpha0 * late**2

    return lr, alpha, 0.239994, 0.635963
