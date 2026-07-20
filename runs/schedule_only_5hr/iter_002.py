"""Iter 002: Cosine with warm restarts + phased alpha.

Changes from iter_001:
- 3 warm restart cycles (cosine annealing with restarts)
- Alpha starts low, ramps up - explore first, then enforce feasibility
- Keep standard Adam betas
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Cosine annealing with warm restarts (3 cycles)
    # Cycle lengths: 1/7, 2/7, 4/7 of total (doubling)
    cycle1 = total_steps // 7
    cycle2 = 2 * total_steps // 7
    cycle3 = total_steps - cycle1 - cycle2

    # Determine which cycle and local progress
    in_c1 = step < cycle1
    in_c2 = (step >= cycle1) & (step < cycle1 + cycle2)

    local_t = jnp.where(
        in_c1, step / jnp.maximum(cycle1, 1.0),
        jnp.where(in_c2,
                   (step - cycle1) / jnp.maximum(cycle2, 1.0),
                   (step - cycle1 - cycle2) / jnp.maximum(cycle3, 1.0)))

    lr_min = 0.01 * lr0
    lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + jnp.cos(jnp.pi * local_t))

    # Alpha: monotonically increases from 0.5*alpha0 to 4*alpha0
    # This lets early iterations explore, late iterations enforce feasibility
    alpha = alpha0 * (0.5 + 3.5 * t)

    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
