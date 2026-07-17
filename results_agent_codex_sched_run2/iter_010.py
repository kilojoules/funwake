"""Seed LR with quadratic penalty ramp.

HYPOTHESIS: The seed inverse LR shape is stronger than monotone linear decay,
but replacing strict inverse alpha coupling with a quadratic ramp can delay
constraint domination long enough to improve wake rearrangement before a firm
late repair phase.
AXIS: alpha_quadratic_ramp on seed inverse LR decay with TopFarm low-momentum
Adam and no LR pulses.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    decay_progress = decay_step / jnp.maximum(decay_steps - 1.0, 1.0)
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    lr = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)

    quad = decay_progress * decay_progress
    alpha_base = alpha0 * (1.0 + 125.0 * quad)

    repair_t = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.70) / 0.30))
    repair = repair_t * repair_t * (3.0 - 2.0 * repair_t)
    alpha = alpha_base * (1.0 + 40.0 * repair)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
