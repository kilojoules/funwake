"""Seed decay with a decaying sinusoidal LR shake.

HYPOTHESIS: A positive-only sinusoidal shake during the decay phase can provide
several controlled wake-layout escapes without the hard restart jumps that
hurt the SGDR attempt.
AXIS: lr_sinusoidal_shake on the seed inverse decay, with inverse LR-alpha
coupling and end-only feasibility repair.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase
    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    progress = decay_step / jnp.maximum(decay_steps - 1.0, 1.0)

    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    lr_base = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)

    sinusoidal = jnp.sin(5.0 * jnp.pi * progress)
    positive_shake = jnp.maximum(0.0, sinusoidal)
    envelope = jnp.exp(-3.1 * progress) * (1.0 - 0.35 * progress)
    active = jnp.where(progress < 0.62, 1.0, 0.0)
    shake = positive_shake * envelope * active

    end_gate_raw = (progress - 0.78) / 0.22
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr = lr_base * (1.0 + 4.4 * shake) * (1.0 - 0.90 * end_gate)

    alpha_base = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(lr, 1e-10),
        alpha0,
    )
    alpha_soften = 1.0 - 0.45 * shake
    alpha_repair = 1.0 + 3.2 * end_gate
    alpha = alpha_base * alpha_soften * alpha_repair

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
