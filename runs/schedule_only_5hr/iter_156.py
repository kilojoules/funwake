"""Iter 156: Two-phase Adam — standard Adam betas for exploration, SGD-like for refinement.

Phase 1 (0-55%): Standard Adam (0.9, 0.999) with high LR — smooth gradients for exploration.
Phase 2 (55-100%): Low beta SGD-like (0.1, 0.2) with low LR — fast convergence.
Cosine LR decay throughout. 5x alpha coupled to 1/lr.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Single cosine LR decay
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # 5x alpha coupled to 1/lr (best from iter_153)
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Phase transition at t=0.55 via smooth sigmoid
    phase = 1.0 / (1.0 + jnp.exp(-30.0 * (t - 0.55)))

    # Phase 1: standard Adam (0.9, 0.999) -> Phase 2: SGD-like (0.1, 0.2)
    beta1 = 0.9 * (1.0 - phase) + 0.1 * phase
    beta2 = 0.999 * (1.0 - phase) + 0.2 * phase

    return lr, alpha, beta1, beta2
