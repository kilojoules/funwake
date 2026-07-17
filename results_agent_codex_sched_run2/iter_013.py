"""Intermediate final repair for the polynomial pulse schedule.

HYPOTHESIS: Attempt 8 made ROWP feasible but paid a small train-AEP cost versus
attempt 7; an intermediate final repair may keep held-out feasibility while
recovering some objective value.
AXIS: schedule_two_phase tradeoff on the polynomial LR pulse: same explore
phase, softened final LR taper and alpha repair.
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

    end_gate_raw = (progress - 0.70) / 0.30
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + 4.0 * one_cycle) * (1.0 - 0.95 * end_gate)

    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr, 1e-10),
        alpha0,
    )
    alpha_dip = 1.0 - 0.76 * one_cycle
    alpha_repair = 1.0 + 6.0 * end_gate
    alpha = alpha_base * alpha_dip * alpha_repair

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
