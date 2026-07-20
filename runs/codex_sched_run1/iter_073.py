"""HYPOTHESIS: A differential-evolution-found dual-bump schedule can improve
over the hand-refined single late bump by using warmup, stronger penalty
coupling, and higher Adam memory to reach a different feasible basin.
AXIS: imported_dual_bump_de_best
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099217214548 * lr0
    lr_min = lr_init / (10.0 ** 3.4204682031708615)

    warmup = 0.06143342573031323
    warmup_lr = lr_init * t / jnp.maximum(warmup, 1e-6)
    cosine_t = (t - warmup) / jnp.maximum(1.0 - warmup, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup, warmup_lr, cosine_lr)

    late_bump = 0.1657844986712761 * lr_init * jnp.exp(
        -0.5 * ((t - 0.9235435629977167) / 0.11281119470891057) ** 2
    )
    mid_bump = 0.4582054854798933 * lr_init * jnp.exp(
        -0.5 * ((t - 0.5461616321072262) / 0.1203474900183133) ** 2
    )
    lr = jnp.maximum(lr_base + late_bump + mid_bump, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 2.87947841945131 * alpha0 * lr_init / lr
    alpha = alpha + 16.85094598409327 * alpha0 * late**2

    return lr, alpha, 0.23999415122738466, 0.6359632116018108
