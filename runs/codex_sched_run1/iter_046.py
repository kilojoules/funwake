"""HYPOTHESIS: After shelf, multi-kick, and asymmetric-tail variants underperformed, the strongest move is to restore the validated centered bump and remove the last attempt's alpha over-stiffening.
AXIS: validated_center_bump_restore
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr = lr_base + 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 3.0 * alpha0 * late**2

    return lr, alpha, 0.3, 0.5
