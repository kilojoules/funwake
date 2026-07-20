"""Iter 176: Cyclic alpha with cosine LR.

Instead of monotonically increasing alpha, oscillate it with an
upward trend. This alternates between 'explore freely' and
'enforce constraints' phases, potentially finding better solutions.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Cosine LR decay with bump at 0.7 (proven good)
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Cyclic alpha: oscillates with 4 cycles, upward trend
    base_alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    cycle = 0.5 * (1.0 + jnp.sin(2.0 * jnp.pi * 4.0 * t))  # 4 cycles
    envelope = 1.0 + 2.0 * t  # grows from 1x to 3x
    alpha = base_alpha * (0.5 + 0.5 * cycle * envelope)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
