"""HYPOTHESIS: The transferred dual-bump LR path is the strongest train
basin, but its inverse-LR alpha coupling relaxes exactly during the two
mobility windows. Small localized alpha counter-ramps may preserve the dual
movement while avoiding the over-stiff partial-decoupling variant.
AXIS: dual_bump_local_alpha_counter_ramps
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
    alpha = alpha + 0.55 * alpha0 * mid_shape + 0.35 * alpha0 * late_shape

    return lr, alpha, 0.239994, 0.635963
