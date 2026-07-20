"""Iter 130: Polynomial decay (power 2) + 6x LR + time-varying betas.

Different from all previous attempts:
- Polynomial (1-t)^2 instead of cosine (slower initial, faster final decay)
- 6x initial LR (vs 4x) for more aggressive exploration
- Beta1: 0.4->0.15, Beta2: 0.6->0.3 (high momentum early, low late)
- lr/8000 min ratio
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 6.0 * lr0
    lr_min = lr_init / 8000.0

    # Polynomial decay (power 2)
    lr = lr_min + (lr_init - lr_min) * (1.0 - t) ** 2

    # Alpha coupled to 1/lr
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Smooth beta transition: high momentum early, low late
    beta1 = 0.4 * (1.0 - t) + 0.15 * t
    beta2 = 0.6 * (1.0 - t) + 0.3 * t

    return lr, alpha, beta1, beta2
