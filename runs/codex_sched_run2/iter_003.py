"""Localized Gaussian exploration with end-only repair.

HYPOTHESIS: The high-scoring seed-style Gaussian bump benefits from minimal
late interference; restricting the feasibility repair to the final fraction
should preserve train AEP while still passing tight polygon checks.
AXIS: lr_gaussian_bumps with alpha_anti_phase_dip and an end-only LR/alpha
repair gate.
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

    bump_center = 0.10
    bump_width = 0.05
    bump_amp = 4.0
    gaussian_bump = jnp.exp(
        -0.5 * ((progress - bump_center) / bump_width) ** 2
    )

    end_gate_raw = (progress - 0.75) / 0.25
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + bump_amp * gaussian_bump) * (1.0 - 0.92 * end_gate)

    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr, 1e-10),
        alpha0,
    )
    alpha_dip = 1.0 - 0.80 * gaussian_bump
    alpha_repair = 1.0 + 4.0 * end_gate
    alpha = alpha_base * alpha_dip * alpha_repair

    beta1 = 0.0
    beta2 = 0.2

    return lr, alpha, beta1, beta2
