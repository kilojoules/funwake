"""HYPOTHESIS: The best LR/alpha path may benefit from slightly less Adam
memory everywhere, making the normalized update track the late constraint
counter-ramp without changing the proven movement timing.
AXIS: best_path_slightly_lower_adam_memory
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump_shape = jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)
    lr = lr_base + 0.3 * lr_init * bump_shape

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 1.18 * alpha0 * bump_shape

    return lr, alpha, 0.28, 0.47
