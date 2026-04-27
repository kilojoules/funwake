"""Seed schedule: coupled penalty-LR (TopFarm-style multiplicative decay).

This matches what topfarm_sgd_solve does internally.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    """Coupled LR-penalty schedule.

    Args:
        step: current iteration (0 to total_steps-1)
        total_steps: total number of iterations
        lr0: initial learning rate
        alpha0: initial penalty weight

    Returns:
        (lr, alpha, beta1, beta2)
    """
    # Constant LR for first 1/3, then decay for remaining 2/3
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase

    # In decay phase: lr = lr0 / (1 + mid * (step - const_phase))
    # Targeting lr_final = 0.01 * lr0
    # So: 0.01 = 1 / (1 + mid * decay_steps) => mid = 99 / decay_steps
    mid = 99.0 / jnp.maximum(decay_steps, 1.0)

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    lr = jnp.where(decaying, lr0 / (1 + mid * decay_step), lr0)

    # Alpha ramps UP as LR decays
    alpha = jnp.where(decaying,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10),
                      alpha0)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
