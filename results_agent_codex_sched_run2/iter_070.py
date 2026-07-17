"""Deterministic two-lobed exploration pulse on the proven repair envelope.

HYPOTHESIS: The best noisy schedule may be benefiting from a coarse mid-run
step-size displacement rather than randomness itself. A smooth deterministic
push-pull pulse can test the same basin move while preserving the alpha repair
path that generalized.
AXIS: deterministic lr_gaussian_bumps replacing lr_noise_injection.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00034 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    early_push = jnp.exp(-0.5 * ((t - 0.30) / 0.075) ** 2)
    mid_release = jnp.exp(-0.5 * ((t - 0.50) / 0.090) ** 2)
    late_quench = jnp.exp(-0.5 * ((t - 0.77) / 0.060) ** 2)
    pulse = 0.105 * early_push - 0.055 * mid_release - 0.030 * late_quench
    lr = jnp.maximum(lr_base * (1.0 + pulse), lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.32 * t + 0.06 * t * t)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.69) / 0.31))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.5 * late + 56.0 * late * late)

    beta1 = 0.09
    beta2 = 0.20
    return lr, alpha, beta1, beta2
