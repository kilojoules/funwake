"""Single triangular one-cycle pulse on seed decay.

HYPOTHESIS: A single early triangular one-cycle pulse can provide the useful
post-grid layout move without the repeated oscillations or hard restarts that
reduced AEP in the last two attempts.
AXIS: lr_one_cycle / lr_cyclical_triangular pulse with inverse alpha coupling
and late feasibility repair.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    progress = decay_step / jnp.maximum(decay_steps - 1.0, 1.0)

    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    lr_base = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)

    rise = (progress - 0.025) / 0.075
    fall = (0.205 - progress) / 0.105
    triangular = jnp.maximum(0.0, jnp.minimum(rise, fall))
    triangular = jnp.minimum(1.0, triangular)
    one_cycle = triangular * triangular * (3.0 - 2.0 * triangular)

    end_gate_raw = (progress - 0.75) / 0.25
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + 4.0 * one_cycle) * (1.0 - 0.92 * end_gate)

    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr, 1e-10),
        alpha0,
    )
    alpha_dip = 1.0 - 0.72 * one_cycle
    alpha_repair = 1.0 + 4.0 * end_gate
    alpha = alpha_base * alpha_dip * alpha_repair

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
