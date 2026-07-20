"""Iter 001: Cosine annealing LR + standard Adam betas + coupled alpha.

Changes from seed:
- Cosine annealing instead of multiplicative decay (smoother, avoids plateau)
- Standard Adam momentum (beta1=0.9) for better gradient smoothing
- Higher beta2=0.999 for better adaptive scaling
- Alpha still coupled inversely to LR
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Cosine annealing: lr decays from lr0 to 0.01*lr0
    lr_min = 0.01 * lr0
    lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + jnp.cos(jnp.pi * t))

    # Alpha inversely coupled to LR (same principle as seed)
    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    # Standard Adam parameters
    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
