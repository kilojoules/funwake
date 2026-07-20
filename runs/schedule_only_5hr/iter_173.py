"""Iter 173: Cosine warm restarts (SGDR-style) with 3 cycles.

Fundamentally different from the single cosine+bump approach.
3 cosine cycles with increasing alpha baseline per cycle.
Each restart lets the optimizer escape local minima.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # 3 warm restart cycles: 40%, 35%, 25% of total steps
    # Cycle boundaries at t=0.4, t=0.75, t=1.0
    c1_end = 0.40
    c2_end = 0.75

    # Determine which cycle we're in and local progress within cycle
    in_c1 = t < c1_end
    in_c2 = (t >= c1_end) & (t < c2_end)
    # in_c3 = t >= c2_end

    t_local = jnp.where(
        in_c1,
        t / c1_end,
        jnp.where(
            in_c2,
            (t - c1_end) / (c2_end - c1_end),
            (t - c2_end) / (1.0 - c2_end)
        )
    )

    # Cosine annealing within each cycle
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local))

    # Alpha increases per cycle: 3x, 5x, 8x base
    alpha_mult = jnp.where(in_c1, 3.0, jnp.where(in_c2, 5.0, 8.0))
    alpha = alpha_mult * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Beta1 decreases per cycle: more momentum early, less late
    beta1 = jnp.where(in_c1, 0.5, jnp.where(in_c2, 0.3, 0.15))
    beta2 = 0.5

    return lr, alpha, beta1, beta2
