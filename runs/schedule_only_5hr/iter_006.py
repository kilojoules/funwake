"""Iter 006: Seed schedule but with moderate Adam betas (0.5, 0.5).

Test whether moderate momentum helps without causing infeasibility.
Same LR/alpha schedule as seed.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    # Exact same LR/alpha as seed
    const_phase = total_steps // 3
    decay_steps = total_steps - const_phase
    mid = 99.0 / jnp.maximum(decay_steps, 1.0)

    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    lr = jnp.where(decaying, lr0 / (1 + mid * decay_step), lr0)

    alpha = jnp.where(decaying,
                      alpha0 * lr0 / jnp.maximum(lr, 1e-10),
                      alpha0)

    # Moderate momentum
    beta1 = 0.5
    beta2 = 0.5

    return lr, alpha, beta1, beta2
