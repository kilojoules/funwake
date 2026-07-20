"""Iter 009: Annealing beta1 (low→high) + seed LR + stronger alpha.

Hypothesis: start with low beta1 for responsive exploration, increase
to high beta1 for smooth convergence. Keep beta2 moderate.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Seed-style LR decay
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase
    mid = 99.0 / jnp.maximum(decay_steps, 1.0)

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    lr = jnp.where(decaying, lr0 / (1 + mid * decay_step), lr0)

    # Alpha: 2x stronger than seed
    alpha = jnp.where(decaying,
                      2.0 * alpha0 * lr0 / jnp.maximum(lr, 1e-10),
                      2.0 * alpha0)

    # Beta1: anneal from 0.1 to 0.7
    beta1 = 0.1 + 0.6 * t
    beta2 = 0.3

    return lr, alpha, beta1, beta2
