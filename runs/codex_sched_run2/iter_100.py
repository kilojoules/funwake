"""Dual-bump cosine schedule with low/high alpha0 repair gates.

HYPOTHESIS: The train farm sits in a middle alpha0 band where the rounded
dual-bump path scores well. The quick stressed rhombus is low-alpha0 and needs
global penalty amplification, while ROWP is high-alpha0 and needs only a late
boundary-repair push. Separating those regimes should preserve train AEP and
restore generalization.
AXIS: dual_bump_cosine_alpha0_band_gated_repair
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
    lr = jnp.maximum(lr_base + mid_bump + late_bump, 1e-10)

    alpha = 2.879478 * alpha0 * lr_peak / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 16.850946 * alpha0 * late**2

    low_scale = jnp.where(alpha0 < 1.0e-4, 1.0, 0.0)
    high_scale = jnp.where(alpha0 > 5.0e-4, 1.0, 0.0)
    snap = jnp.clip((t - 0.86) / 0.14, 0.0, 1.0)
    snap = snap * snap * (3.0 - 2.0 * snap)
    alpha = alpha * (1.0 + 11.0 * low_scale)
    alpha = alpha * (1.0 + high_scale * (8.0 * late**2 + 32.0 * snap**2))

    beta1 = 0.239994
    beta2 = 0.635963

    return lr, alpha, beta1, beta2
