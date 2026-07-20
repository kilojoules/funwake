"""Iter 256: SGDR warm restarts — 3 cosine cycles with diminishing peaks.

Each cycle: cosine from peak -> min, then snap back up.
Peaks decay: 100%, 50%, 25% of lr_init.
Cycles: [0, 0.4), [0.4, 0.7), [0.7, 1.0)
Alpha ramps with inverse-lr coupling + quadratic tail.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cycle boundaries and peak multipliers
    # Cycle 1: [0, 0.4) peak=1.0
    # Cycle 2: [0.4, 0.7) peak=0.5
    # Cycle 3: [0.7, 1.0) peak=0.25
    c1_end = 0.4
    c2_end = 0.7

    # Cycle-local progress (0 to 1 within each cycle)
    in_c1 = t < c1_end
    in_c2 = (t >= c1_end) & (t < c2_end)

    t_c1 = t / c1_end
    t_c2 = (t - c1_end) / (c2_end - c1_end)
    t_c3 = (t - c2_end) / (1.0 - c2_end)

    # Cosine decay within each cycle
    lr_c1 = lr_min + (1.0 * lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_c1))
    lr_c2 = lr_min + (0.5 * lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_c2))
    lr_c3 = lr_min + (0.25 * lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_c3))

    lr = jnp.where(in_c1, lr_c1, jnp.where(in_c2, lr_c2, lr_c3))

    # Alpha: 5x coupled + quadratic tail from t=0.5
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
