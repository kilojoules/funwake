"""Iter 069: Single cosine decay (no warm restarts) + 4x LR.

Higher initial LR with smooth cosine decay from the start.
No constant phase — cosine decay over full 8000 steps.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 30000.0

    # Full cosine decay
    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Alpha coupled to 1/lr
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
