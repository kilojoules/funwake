"""Moderate-momentum variant of the best feasible pulse schedule.

HYPOTHESIS: The best prior schedule's LR pulse finds a better wake basin, but
very low Adam moments may leave late repair noisy; moderate moments can smooth
the final approach without changing the explored LR/alpha timing.
AXIS: adam_topfarm_low-to-moderate momentum change on the prior polynomial
pulse schedule, retaining inverse alpha coupling and strong final repair.
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

    u = (progress - 0.10) / 0.13
    compact = jnp.maximum(0.0, 1.0 - u * u)
    one_cycle = compact * compact

    end_gate_raw = (progress - 0.68) / 0.32
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + 4.0 * one_cycle) * (1.0 - 0.96 * end_gate)

    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr, 1e-10),
        alpha0,
    )
    alpha_dip = 1.0 - 0.76 * one_cycle
    alpha_repair = 1.0 + 8.0 * end_gate
    alpha = alpha_base * alpha_dip * alpha_repair

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
