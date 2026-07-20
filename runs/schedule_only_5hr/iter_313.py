"""Iter 313: Piecewise linear schedule — 5 segments, no cosine.

Maximally simple: linear LR segments with sharp transitions.
25% constant → 25% slow decay → 15% steep decay → 10% bump → 25% final decay.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Piecewise linear LR
    # [0, 0.25): constant at lr_init
    # [0.25, 0.50): decay to 0.4 * lr_init
    # [0.50, 0.65): decay to 0.1 * lr_init
    # [0.65, 0.75): bump back up to 0.3 * lr_init
    # [0.75, 1.0): decay to lr_min

    seg1 = lr_init  # constant
    seg2_t = jnp.clip((t - 0.25) / 0.25, 0.0, 1.0)
    seg2 = lr_init * (1.0 - 0.6 * seg2_t)  # lr_init → 0.4*lr_init
    seg3_t = jnp.clip((t - 0.50) / 0.15, 0.0, 1.0)
    seg3 = 0.4 * lr_init * (1.0 - 0.75 * seg3_t)  # 0.4 → 0.1
    seg4_t = jnp.clip((t - 0.65) / 0.10, 0.0, 1.0)
    seg4 = 0.1 * lr_init + 0.2 * lr_init * jnp.sin(jnp.pi * seg4_t)  # bump
    seg5_t = jnp.clip((t - 0.75) / 0.25, 0.0, 1.0)
    seg5 = 0.1 * lr_init * (1.0 - seg5_t) + lr_min * seg5_t

    lr = jnp.where(t < 0.25, seg1,
         jnp.where(t < 0.50, seg2,
         jnp.where(t < 0.65, seg3,
         jnp.where(t < 0.75, seg4, seg5))))

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
