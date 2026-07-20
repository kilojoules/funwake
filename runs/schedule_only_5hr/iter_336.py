"""Iter 336: Triangular cyclic LR (Smith 2017) + 326-style alpha.

Triangular wave cycles the LR up and down. 4 cycles over the run.
Each successive cycle has a lower ceiling (1-cycle decay).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # 4 triangular cycles
    n_cycles = 4.0
    cycle_t = jnp.mod(t * n_cycles, 1.0)  # 0->1 within each cycle
    # Triangle: goes 0->1->0 within each cycle
    triangle = 1.0 - jnp.abs(2.0 * cycle_t - 1.0)
    # Decaying envelope
    envelope = 1.0 - 0.7 * t  # peak decays from 1.0 to 0.3
    lr = lr_min + (lr_init - lr_min) * triangle * envelope

    # Alpha: coupled + late enforcement
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 4.0 * alpha0 * late ** 2
    fb1 = 15.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.84) / 0.02) ** 2)
    fb2 = 25.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.92) / 0.015) ** 2)
    alpha = alpha_base + alpha_extra + fb1 + fb2

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
