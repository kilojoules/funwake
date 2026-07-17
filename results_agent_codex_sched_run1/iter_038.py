"""HYPOTHESIS: The broad dual-bump schedule needs the late bump for polishing mobility, but alpha should mostly ignore that late bump so boundary pressure remains high.
AXIS: broad_dual_bump_late_alpha_guard
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    mid_bump = 0.458205 * lr_init * jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_bump = 0.165784 * lr_init * jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr = jnp.maximum(lr_base + mid_bump + late_bump, 1e-10)

    alpha_lr = jnp.maximum(lr_base + mid_bump + 0.25 * late_bump, 1e-10)
    alpha = 2.879478 * alpha0 * lr_init / alpha_lr
    alpha = alpha + 16.850946 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2

    return lr, alpha, 0.239994, 0.635963
