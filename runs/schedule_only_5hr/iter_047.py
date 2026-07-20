"""Iter 047: Two-stage exploration: 15% at 5x LR, 20% at 3x LR, then decay.

Super-exploration burst first, then settle into good region, then enforce.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    phase1_end = 0.15  # super-explore
    phase2_end = 0.35  # normal explore

    in_p1 = t < phase1_end
    in_p2 = (t >= phase1_end) & (t < phase2_end)

    lr_init = 3.0 * lr0
    decay_t = jnp.maximum(t - phase2_end, 0.0) / (1.0 - phase2_end)
    lr_decay = lr_init / (1 + 29999.0 * decay_t)

    lr = jnp.where(in_p1, 5.0 * lr0,
         jnp.where(in_p2, 3.0 * lr0, lr_decay))

    alpha = jnp.where(t < phase2_end, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
