"""Iter 333: 5x initial LR with cosine decay + moderate alpha.

Hypothesis: higher LR gives more exploration power. Previous best
approaches used 4x. Try 5x with standard cosine and moderate feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.2
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr = jnp.where(t < plateau_end, lr_init, cosine_lr)

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    fb1 = 15.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    fb2 = 25.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)
    alpha = alpha_base + alpha_extra + fb1 + fb2

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
