"""HYPOTHESIS: A staged schedule with an early LR plateau, two wake-mobility kicks, and late feasibility bursts may escape the single-bump basin while still ending with strong constraints.
AXIS: plateau_multi_kick_feasibility_bursts
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.22
    decay_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    kick1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    kick2 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.78) / 0.03) ** 2)
    settle1 = 0.12 * lr_init * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    settle2 = 0.08 * lr_init * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)
    lr = lr_base + kick1 + kick2 + settle1 + settle2

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 15.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    alpha = alpha + 25.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)

    return lr, alpha, 0.3, 0.5
