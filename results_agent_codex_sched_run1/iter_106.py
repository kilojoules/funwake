"""HYPOTHESIS: Since the last restored dual-bump schedule has plateaued, switch
back to the distinct single late wake-mobility bump family. Keep the proven
cosine backbone, but use a slightly softer 1.18x local constraint counter-ramp
so the bump can move turbines without losing feasibility.
AXIS: single_late_bump_soft_counter_ramp_return
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump_shape = jnp.exp(-0.5 * ((t - 0.692) / 0.050) ** 2)
    lr = lr_base + 0.30 * lr_init * bump_shape
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 5.0 * alpha0 * lr_init / lr
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 1.18 * alpha0 * bump_shape

    return lr, alpha, 0.3, 0.5
