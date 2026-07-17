"""Dual-bump cosine schedule with local alpha softening.

HYPOTHESIS: The soft terminal taper failed the stressed-boundary test, so the
tail needs full repair authority. Keep the non-exponential warmup/cosine
dual-bump path, but give the objective a small advantage only during the two
mobility windows while leaving the final inverse-LR and quadratic penalty
recovery intact.
AXIS: dual_bump_cosine_with_local_alpha_softening
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

    late_shape = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    mid_shape = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    lr = lr_base + 0.165784 * lr_peak * late_shape + 0.458205 * lr_peak * mid_shape
    lr = jnp.maximum(lr, 1e-10)

    alpha = 2.879478 * alpha0 * lr_peak / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 16.850946 * alpha0 * late**2
    alpha = alpha * (1.0 - 0.06 * mid_shape - 0.025 * late_shape)

    beta1 = 0.239994
    beta2 = 0.635963

    return lr, alpha, beta1, beta2
