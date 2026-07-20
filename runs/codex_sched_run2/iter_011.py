"""Short warmup into seed-style inverse decay.

HYPOTHESIS: A short warmup can reduce early Adam transients from the grid
initialization while a slightly higher plateau preserves the useful basin
movement of the seed inverse-decay schedule.
AXIS: warmup plus alpha_coupled_inverse_lr on seed-style inverse LR decay,
with moderate late feasibility repair and no LR pulses.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    warm = 0.06
    warm_t = jnp.minimum(1.0, t / warm)
    warm_t = warm_t * warm_t * (3.0 - 2.0 * warm_t)

    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase
    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)

    lr_peak = 1.12 * lr0
    lr_warm = lr0 * (0.25 + 0.87 * warm_t)

    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    lr_decay = lr_peak / (1.0 + mid * decay_step)
    lr_body = jnp.where(decaying, lr_decay, lr_peak)
    lr = jnp.where(t < warm, lr_warm, lr_body)

    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    repair_t = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.72) / 0.28))
    repair = repair_t * repair_t * (3.0 - 2.0 * repair_t)
    alpha = alpha * (1.0 + 25.0 * repair)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
