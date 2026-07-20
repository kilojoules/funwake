"""Iter 291: SGDR warm restarts with escalating alpha per cycle.

3 cosine cycles with decreasing amplitude. Each cycle starts with a
brief LR spike and decays. Alpha escalates cycle-over-cycle so the
final cycle heavily enforces feasibility.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # 3 cycles: [0, 0.4), [0.4, 0.7), [0.7, 1.0)
    cycle_bounds = jnp.array([0.0, 0.4, 0.7, 1.0])
    # Amplitude decreases each cycle
    amplitudes = jnp.array([1.0, 0.5, 0.2])

    # Determine which cycle we're in and local progress
    in_c0 = t < 0.4
    in_c1 = (t >= 0.4) & (t < 0.7)
    # in_c2 = t >= 0.7

    local_t = jnp.where(in_c0, t / 0.4,
              jnp.where(in_c1, (t - 0.4) / 0.3,
                        (t - 0.7) / 0.3))
    local_t = jnp.clip(local_t, 0.0, 1.0)

    amp = jnp.where(in_c0, 1.0,
          jnp.where(in_c1, 0.5, 0.2))

    # Cosine decay within each cycle
    lr = lr_min + amp * (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * local_t))

    # Alpha: base inverse coupling + cycle escalation
    cycle_alpha_mult = jnp.where(in_c0, 1.0,
                       jnp.where(in_c1, 3.0, 10.0))
    alpha = cycle_alpha_mult * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Beta scheduling: high momentum early, lower in final cycle for precision
    beta1 = jnp.where(in_c0, 0.5,
            jnp.where(in_c1, 0.3, 0.15))
    beta2 = jnp.where(in_c0, 0.7,
            jnp.where(in_c1, 0.5, 0.3))

    return lr, alpha, beta1, beta2
