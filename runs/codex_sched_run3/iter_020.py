"""HYPOTHESIS: Triangular cyclical LR pulses can approximate the useful
escape timing of bump schedules while using sharper, bounded exploration
windows that spend less time disturbing constraint repair.
AXIS: lr_cyclical_triangular with inverse-LR alpha coupling and final repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    base = 0.00115 + 0.90 * (1.0 - t) * (1.0 - t)

    triangular_1 = 3.20 * jnp.clip(1.0 - jnp.abs(t - 0.105) / 0.105, 0.0, 1.0)
    triangular_2 = 2.42 * jnp.clip(1.0 - jnp.abs(t - 0.320) / 0.120, 0.0, 1.0)
    triangular_3 = 1.22 * jnp.clip(1.0 - jnp.abs(t - 0.545) / 0.110, 0.0, 1.0)
    triangular_4 = 0.48 * jnp.clip(1.0 - jnp.abs(t - 0.730) / 0.075, 0.0, 1.0)
    cyclical_pulses = triangular_1 + triangular_2 + triangular_3 + triangular_4

    lr = lr0 * (base + cyclical_pulses)

    late = jnp.clip((t - 0.61) / 0.39, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.865) / 0.135, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    lr_safe = jnp.maximum(lr + 0.040 * lr0, 1e-10)
    alpha = alpha0 * 3.22 * lr0 / lr_safe
    pulse_norm = cyclical_pulses / (1.0 + cyclical_pulses)
    alpha = alpha * (1.0 - 0.63 * (1.0 - 0.47 * t) * pulse_norm)
    alpha = alpha * (1.0 + 2.85 * t + 22.0 * late * late + 162.0 * tail * tail)
    alpha = alpha * (1.0 + 66.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00107 - 0.00064 * squeeze), lr)

    beta1 = 0.09 + 0.105 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
