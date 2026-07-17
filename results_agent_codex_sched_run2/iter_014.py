"""Warmup graft on the best feasible polynomial pulse schedule.

HYPOTHESIS: The attempt-8 pulse/repair timing is the best feasible shape so
far; adding a very short warmup can reduce early Adam bias-correction shock
without changing the basin-changing pulse later in the decay phase.
AXIS: warmup added to the prior polynomial pulse schedule while preserving
inverse alpha coupling and strong final repair.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    progress = decay_step / jnp.maximum(decay_steps - 1.0, 1.0)
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    lr_seed = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)

    warm_t = jnp.minimum(1.0, t / 0.045)
    warm_t = warm_t * warm_t * (3.0 - 2.0 * warm_t)
    lr_warm = lr0 * (0.35 + 0.65 * warm_t)
    lr_base = jnp.where(t < 0.045, lr_warm, lr_seed)

    u = (progress - 0.10) / 0.13
    compact = jnp.maximum(0.0, 1.0 - u * u)
    one_cycle = compact * compact

    end_gate_raw = (progress - 0.68) / 0.32
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + 4.0 * one_cycle) * (1.0 - 0.96 * end_gate)

    alpha_base = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    alpha_dip = 1.0 - 0.76 * one_cycle
    alpha_repair = 1.0 + 8.0 * end_gate
    alpha = alpha_base * alpha_dip * alpha_repair

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
