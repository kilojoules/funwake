"""Iter 276: Two-segment cosine with mid-point restart.

Segment 1 (0-50%): cosine from lr_init to 0.1*lr_init — explore
Segment 2 (50-100%): cosine from 0.3*lr_init to lr_min — refine
A structured restart that gives two full cosine cycles.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    seg1_end = 0.5
    in_seg1 = t < seg1_end

    # Segment 1: cosine from lr_init to 0.1*lr_init
    t1 = t / seg1_end
    lr1_min = 0.1 * lr_init
    lr1 = lr1_min + (lr_init - lr1_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t1))

    # Segment 2: cosine from 0.3*lr_init to lr_min
    t2 = (t - seg1_end) / (1.0 - seg1_end)
    lr2_peak = 0.3 * lr_init
    lr2 = lr_min + (lr2_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t2))

    lr = jnp.where(in_seg1, lr1, lr2)

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
