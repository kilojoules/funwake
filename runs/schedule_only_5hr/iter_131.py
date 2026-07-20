"""Iter 131: 3-phase schedule with phase-specific betas.

Phase 1 (0-25%): High LR, low alpha, low beta1 → free exploration
Phase 2 (25-65%): Cosine-decaying LR, ramping alpha, moderate beta1 → refinement
Phase 3 (65-100%): Low LR, high alpha, higher beta1 → convergence + feasibility
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.5 * lr0
    lr_min = lr_init / 10000.0

    # Phase boundaries
    p1_end = 0.25
    p2_end = 0.65

    in_p1 = t < p1_end
    in_p2 = (t >= p1_end) & (t < p2_end)

    # Phase 1: constant high LR
    lr_p1 = lr_init

    # Phase 2: cosine decay from lr_init to a mid value
    p2_t = (t - p1_end) / (p2_end - p1_end)
    lr_mid = lr_init * 0.05
    lr_p2 = lr_mid + (lr_init - lr_mid) * 0.5 * (1.0 + jnp.cos(jnp.pi * p2_t))

    # Phase 3: cosine decay from mid to min
    p3_t = (t - p2_end) / (1.0 - p2_end)
    lr_p3 = lr_min + (lr_mid - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * p3_t))

    lr = jnp.where(in_p1, lr_p1, jnp.where(in_p2, lr_p2, lr_p3))

    # Alpha: low in phase 1, coupled in phases 2-3
    alpha_p1 = 0.5 * alpha0
    alpha_coupled = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = jnp.where(in_p1, alpha_p1, alpha_coupled)

    # Beta1: low→moderate→moderate-high
    beta1 = jnp.where(in_p1, 0.15, jnp.where(in_p2, 0.3, 0.45))
    beta2 = 0.5

    return lr, alpha, beta1, beta2
