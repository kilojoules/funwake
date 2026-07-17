"""Smooth LR with mid-run constraint-pressure cycling.

HYPOTHESIS: Instead of moving the layout basin through noisy learning-rate
steps, alternating slightly softer and stronger constraint pressure during the
middle search phase may let wake gains form and then repair before polishing.
AXIS: alpha_anti_phase_dip/cyclic alpha on deterministic exponential LR.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.08 * lr0
    lr_floor = 0.00034 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-7.92 * t * t), lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.34 * t)

    open_gate = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.17) / 0.13))
    close_gate = jnp.minimum(1.0, jnp.maximum(0.0, (0.66 - t) / 0.16))
    search_gate = open_gate * open_gate * (3.0 - 2.0 * open_gate)
    search_gate = search_gate * close_gate * close_gate * (3.0 - 2.0 * close_gate)
    cycle = jnp.sin(7.0 * jnp.pi * t + 0.35)
    alpha = alpha * (1.0 - 0.115 * search_gate * cycle)

    soft = jnp.exp(-0.5 * ((t - 0.43) / 0.11) ** 2)
    alpha = alpha * (1.0 - 0.075 * soft)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.2 * late + 54.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr_base, alpha, beta1, beta2
