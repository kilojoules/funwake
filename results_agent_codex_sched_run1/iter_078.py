"""HYPOTHESIS: The best LR/alpha trajectory may be correct, but the final
settling phase can benefit from more Adam memory after the mobility bump,
reducing small post-bump oscillations without adding new movement.
AXIS: post_bump_terminal_adam_smoothing
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

    settle = jnp.clip((t - 0.76) / 0.14, 0.0, 1.0)
    smooth = settle * settle * (3.0 - 2.0 * settle)
    beta1 = 0.3 + 0.18 * smooth
    beta2 = 0.5 + 0.18 * smooth

    return lr, alpha, beta1, beta2
