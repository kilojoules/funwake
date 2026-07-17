"""HYPOTHESIS: The best alpha counter-ramp may pair with a slightly lower,
broader LR mobility pulse so turbines keep useful movement without overshooting
the final feasible basin.
AXIS: lr_bump_broader_softer_with_best_alpha
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr_bump = jnp.exp(-0.5 * ((t - 0.692) / 0.054) ** 2)
    lr = lr_base + 0.27 * lr_init * lr_bump

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 1.18 * alpha0 * lr_bump

    return lr, alpha, 0.3, 0.5
