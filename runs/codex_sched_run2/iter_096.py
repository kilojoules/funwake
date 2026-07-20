"""Rounded dual-bump cosine schedule with soft terminal taper.

HYPOTHESIS: The stable 5564.5 schedules are trapped by the noisy exponential
path, while the constant dual-kick variant over-stays at the base stride. A
smooth warmup/cosine path with one broad mid-run bump and one late bump gives
two controlled basin moves, then tapers only the final tail to reduce
penalty-dominated drift without disabling repair.
AXIS: warmup_cosine_dual_bump_soft_tail_with_inverse_penalty
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    lr_peak = 4.327099 * lr0
    lr_min = lr_peak / (10.0 ** 3.420468)
    warm_end = 0.061433

    warm_lr = lr_peak * t / jnp.maximum(warm_end, 1e-6)
    cos_t = (t - warm_end) / jnp.maximum(1.0 - warm_end, 1e-6)
    cos_lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * cos_t))
    lr_base = jnp.where(t < warm_end, warm_lr, cos_lr)

    mid_bump = 0.458205 * lr_peak * jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_bump = 0.165784 * lr_peak * jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr_active = jnp.maximum(lr_base + mid_bump + late_bump, 1e-10)

    tail = jnp.clip((t - 0.982) / 0.018, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    lr = lr_active * (1.0 - 0.55 * tail)

    alpha = 2.879478 * alpha0 * lr_peak / lr_active
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 16.850946 * alpha0 * late**2

    beta1 = 0.239994
    beta2 = 0.635963

    return lr, alpha, beta1, beta2
