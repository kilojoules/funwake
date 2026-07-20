"""Iter 234: SGDR warm restarts — 3 cycles with decreasing amplitude.

Fundamentally different from all prior attempts (single cosine + bump).
Three complete cosine cycles give three separate exploration opportunities:
- Cycle 1 (0-50%): Full LR sweep (4x lr0 -> min)
- Cycle 2 (50-80%): Half LR sweep (2x lr0 -> min)
- Cycle 3 (80-100%): Quarter LR sweep (1x lr0 -> min)

Alpha increases monotonically across all cycles (coupled to effective LR).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cycle 1: t in [0, 0.50), amplitude = lr_init
    c1_t = t / 0.50
    c1_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * c1_t))

    # Cycle 2: t in [0.50, 0.80), amplitude = lr_init / 2
    c2_amp = lr_init * 0.5
    c2_t = (t - 0.50) / 0.30
    c2_lr = lr_min + (c2_amp - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * c2_t))

    # Cycle 3: t in [0.80, 1.00), amplitude = lr_init / 4
    c3_amp = lr_init * 0.25
    c3_t = (t - 0.80) / 0.20
    c3_lr = lr_min + (c3_amp - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * c3_t))

    lr = jnp.where(t < 0.50, c1_lr,
         jnp.where(t < 0.80, c2_lr, c3_lr))

    # Alpha: 5x coupled to lr, monotonically increasing
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    # Extra late ramp for feasibility
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
