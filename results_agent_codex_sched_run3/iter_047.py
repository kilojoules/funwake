"""HYPOTHESIS: A small constant physical step can avoid the late freeze of the
best decaying schedules, using a smooth penalty ramp to convert sustained
mobility into feasible wake polishing.
AXIS: lr_constant with monotone penalty tightening and medium Adam memory.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0
    lr = 0.020 * lr

    smooth = t * t * (3.0 - 2.0 * t)
    late = jnp.clip((t - 0.58) / 0.42, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.86) / 0.14, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)

    alpha = alpha0 * (18.0 + 80.0 * smooth + 800.0 * late * late)
    alpha = alpha * (1.0 + 120.0 * tail * tail)

    beta1 = (18.0 + 8.0 * smooth) / 100.0
    beta2 = (48.0 + 16.0 * smooth) / 100.0

    return lr, alpha, beta1, beta2
