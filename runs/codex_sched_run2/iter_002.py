"""Seed decay with an early Gaussian LR bump and late repair.

HYPOTHESIS: A short LR bump early in the decay phase can cross wake-layout
barriers after the grid has partially organized, while a paired alpha dip keeps
constraints from freezing that exploration too soon.
AXIS: lr_gaussian_bumps plus alpha_anti_phase_dip, zero first momentum, and
late feasibility repair.
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
    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr_base, 1e-10),
        alpha0,
    )

    bump_center = 0.10
    bump_width = 0.05
    bump_amp = 4.0
    gaussian_bump = jnp.exp(
        -0.5 * ((progress - bump_center) / bump_width) ** 2
    )

    late_taper = 1.0 - 0.92 * progress**4
    lr = lr_base * (1.0 + bump_amp * gaussian_bump) * late_taper

    alpha_dip = 1.0 - 0.50 * gaussian_bump
    repair = 1.0 + 2.0 * progress * progress
    alpha = (
        alpha0
        * lr0
        / jnp.maximum(lr, 1e-10)
        * alpha_dip
        * jnp.where(decaying, repair, 1.0)
    )

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
