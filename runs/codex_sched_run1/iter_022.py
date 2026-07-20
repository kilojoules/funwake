"""HYPOTHESIS: A wider but softer bump can maintain mobility through the productive late window without the objective loss of a stronger peak.
AXIS: cosine_bump_width_sweep
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr = lr_base + 0.28 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.065) ** 2)
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2
    return lr, alpha, 0.3, 0.5
