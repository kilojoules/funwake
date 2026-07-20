"""Iter 003: Three-phase schedule with TopFarm-style low betas.

Phase 1 (0-25%): High LR, low alpha - explore layout space freely
Phase 2 (25-75%): Cosine-decaying LR, ramping alpha - optimize with constraints
Phase 3 (75-100%): Low LR, high alpha - fine-tune feasibility

Uses TopFarm-style low betas (0.1, 0.2) which may work better for
this specific problem (less momentum = more responsive to local gradients).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Phase boundaries
    p1_end = 0.25
    p2_end = 0.75

    in_p1 = t < p1_end
    in_p2 = (t >= p1_end) & (t < p2_end)

    # Phase 1: constant high LR
    # Phase 2: cosine decay from lr0 to 0.05*lr0
    # Phase 3: linear decay from 0.05*lr0 to 0.005*lr0
    p2_t = (t - p1_end) / (p2_end - p1_end)
    p3_t = (t - p2_end) / (1.0 - p2_end)

    lr_p2 = 0.05 * lr0 + 0.5 * (lr0 - 0.05 * lr0) * (1 + jnp.cos(jnp.pi * p2_t))
    lr_p3 = 0.05 * lr0 * (1 - 0.9 * p3_t)

    lr = jnp.where(in_p1, lr0,
         jnp.where(in_p2, lr_p2, lr_p3))

    # Alpha schedule: low -> medium -> high
    alpha = jnp.where(in_p1, 0.3 * alpha0,
            jnp.where(in_p2, alpha0 * (0.3 + 2.7 * p2_t),
                       alpha0 * (3.0 + 7.0 * p3_t)))

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
