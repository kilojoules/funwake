"""Iter 067: Cosine annealing with warm restarts (SGDR-style).

3 cycles of cosine decay with decreasing max LR per cycle.
Each restart lets the optimizer escape local minima.
Alpha coupled to 1/lr for feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_max = 3.0 * lr0
    lr_min = lr_max / 30000.0

    # 3 cycles: lengths 0.2, 0.3, 0.5 of total steps
    # Max LR decays each cycle: 1.0, 0.5, 0.25
    cycle1_end = 0.20
    cycle2_end = 0.50

    # Which cycle are we in?
    in_c1 = t < cycle1_end
    in_c2 = (t >= cycle1_end) & (t < cycle2_end)

    # Cycle progress (0 to 1 within each cycle)
    t_c1 = t / cycle1_end
    t_c2 = (t - cycle1_end) / (cycle2_end - cycle1_end)
    t_c3 = (t - cycle2_end) / (1.0 - cycle2_end)

    # Cosine decay within each cycle
    cos_c1 = 0.5 * (1.0 + jnp.cos(jnp.pi * t_c1))
    cos_c2 = 0.5 * (1.0 + jnp.cos(jnp.pi * t_c2))
    cos_c3 = 0.5 * (1.0 + jnp.cos(jnp.pi * t_c3))

    # LR per cycle (decreasing max)
    lr_c1 = lr_min + (lr_max - lr_min) * cos_c1
    lr_c2 = lr_min + (0.5 * lr_max - lr_min) * cos_c2
    lr_c3 = lr_min + (0.25 * lr_max - lr_min) * cos_c3

    lr = jnp.where(in_c1, lr_c1, jnp.where(in_c2, lr_c2, lr_c3))

    # Alpha coupled to 1/lr
    alpha = alpha0 * lr_max / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
