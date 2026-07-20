"""Iter 308: Cyclic cosine — 2 full cycles within a decaying envelope.

Different from warm restarts: the envelope itself is a cosine decay,
while an inner oscillation creates 2 exploration-convergence cycles.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Decaying envelope: cosine
    envelope = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    # Inner oscillation: 2 cycles, amplitude shrinks with envelope
    oscillation = 0.3 * (envelope - lr_min) * 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * 2.0 * t))
    lr = envelope + oscillation

    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
