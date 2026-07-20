"""HYPOTHESIS: A quadratic late alpha lift can repair the boundary residual left by plain cosine decay without suppressing early wake-driven movement.
AXIS: alpha_quadratic_ramp layered on cosine LR decay with inverse-LR coupling.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    hold_frac = 0.25
    min_lr_frac = 0.005

    phase = jnp.clip((t - hold_frac) / (1.0 - hold_frac), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * phase))
    lr_frac = min_lr_frac + (1.0 - min_lr_frac) * cosine
    lr = lr0 * jnp.where(t < hold_frac, 1.0, lr_frac)

    alpha_base = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    alpha = alpha_base * (1.0 + 1.5 * t * t)

    return lr, alpha, 0.10, 0.20
