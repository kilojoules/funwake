"""Iter 132: SGDR warm restarts - 3 cycles with decaying envelope.

Cycles: [0-2000], [2000-6000], [6000-8000]. Envelope decays max LR per cycle.
Each restart helps escape local optima. beta1=0.3, beta2=0.5.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    s = step * 1.0
    cycle1_end = 2000.0
    cycle2_end = 6000.0

    in_cycle1 = s < cycle1_end
    in_cycle2 = (s >= cycle1_end) & (s < cycle2_end)

    local_t = jnp.where(in_cycle1, s / cycle1_end,
              jnp.where(in_cycle2, (s - cycle1_end) / (cycle2_end - cycle1_end),
                        (s - cycle2_end) / (total_steps - cycle2_end)))

    cycle_max = jnp.where(in_cycle1, lr_init,
                jnp.where(in_cycle2, 0.6 * lr_init, 0.3 * lr_init))

    lr = lr_min + (cycle_max - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * local_t))

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
