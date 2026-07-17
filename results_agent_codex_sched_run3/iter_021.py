"""HYPOTHESIS: A decaying sinusoidal shake can provide repeated basin escapes
without the sharp discontinuities that hurt the triangular pulse attempt.
AXIS: lr_sinusoidal_shake with exponential decay, inverse-LR alpha coupling,
and late feasibility repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    floor = 0.00112 + 0.0068 * (1.0 - t)
    exp_decay = 0.82 * jnp.exp(-3.05 * t)
    phase = 7.5 * jnp.pi * t - 0.68
    shake_raw = 0.5 + 0.5 * jnp.sin(phase)
    shake_env = 3.65 * jnp.exp(-2.15 * t) * (1.0 - 0.16 * t)
    shake = shake_env * shake_raw
    lr = lr0 * (floor + exp_decay + shake)

    late = jnp.clip((t - 0.61) / 0.39, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.865) / 0.135, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    lr_safe = jnp.maximum(lr + 0.041 * lr0, 1e-10)
    alpha = alpha0 * 3.16 * lr0 / lr_safe
    shake_norm = shake / (1.0 + shake)
    alpha = alpha * (1.0 - 0.61 * (1.0 - 0.49 * t) * shake_norm)
    alpha = alpha * (1.0 + 2.82 * t + 21.0 * late * late + 154.0 * tail * tail)
    alpha = alpha * (1.0 + 60.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00108 - 0.00064 * squeeze), lr)

    beta1 = 0.09 + 0.105 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
