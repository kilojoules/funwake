"""Iter 218: SGDR warm restarts (3 cycles) with phase-dependent betas.

Completely different from single-cosine. Three cosine cycles with
decreasing max LR (5x, 2.5x, 1x). Each restart escapes local optima.
Betas: high momentum early (0.7/0.9), low late (0.2/0.4).
Alpha resets partially at each restart to allow repositioning.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # 3 cycles: [0, 0.25), [0.25, 0.55), [0.55, 1.0)
    lr_min = lr0 / 200.0

    in_c0 = t < 0.25
    in_c1 = (t >= 0.25) & (t < 0.55)

    t_local = jnp.where(in_c0, t / 0.25,
              jnp.where(in_c1, (t - 0.25) / 0.3,
                        (t - 0.55) / 0.45))

    max_lr = jnp.where(in_c0, 5.0 * lr0,
             jnp.where(in_c1, 2.5 * lr0,
                       1.0 * lr0))

    # Cosine annealing within each cycle
    lr = lr_min + (max_lr - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local))

    # Alpha: coupled to 1/lr, moderate base
    alpha_base = 4.0 * alpha0 * (5.0 * lr0) / jnp.maximum(lr, 1e-10)
    alpha = jnp.minimum(alpha_base, 40.0 * alpha0)
    # Extra penalty in last 25%
    late = jnp.maximum(t - 0.75, 0.0) / 0.25
    alpha = alpha + 5.0 * alpha0 * late ** 2

    # Phase betas: high momentum early, low late
    beta1 = jnp.where(in_c0, 0.7,
            jnp.where(in_c1, 0.4, 0.2))
    beta2 = jnp.where(in_c0, 0.9,
            jnp.where(in_c1, 0.6, 0.4))

    return lr, alpha, beta1, beta2
