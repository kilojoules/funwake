"""Iter 257: Polynomial (1-t)^3 decay with time-varying betas.

Higher-order polynomial stays high longer, drops faster at end.
beta1 starts at 0.5 (more momentum for exploration) and decays
to 0.1 (less momentum for fine-tuning). beta2 rises from 0.3 to 0.999.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Polynomial decay: (1-t)^3
    lr = lr_min + (lr_init - lr_min) * (1.0 - t) ** 3

    # Alpha: 5x coupled + quadratic tail
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    # Time-varying betas
    # Early: higher momentum (0.5) for exploration
    # Late: lower momentum (0.1) for fine-grained positioning
    beta1 = 0.5 - 0.4 * t
    # Early: lower beta2 (0.3) = more responsive adaptive scaling
    # Late: higher beta2 (0.999) = smoother, more stable
    beta2 = 0.3 + 0.699 * t

    return lr, alpha, beta1, beta2
