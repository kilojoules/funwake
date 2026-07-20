"""HYPOTHESIS: SGDR-style short warm restarts can escape early wake basins,
then a single long annealing cycle plus smooth late penalty repair can recover
feasibility without the previous polynomial tail.
AXIS: lr_sgdr_warm_restarts with anti-phase cyclic alpha and low Adam momentum.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    short_len = total_steps / 10.0
    final_start = 5.0 * short_len
    in_final = step >= final_start

    cycle_start = jnp.where(in_final, final_start,
                            jnp.floor(step / short_len) * short_len)
    cycle_len = jnp.where(in_final, total_steps - final_start, short_len)
    t_cycle = (step - cycle_start) / jnp.maximum(cycle_len - 1.0, 1.0)

    n_cycles = 6.0
    warm_restart_phase = t_cycle

    peak = lr0 * (4.7 - 3.5 * t)
    floor = lr0 * (0.0020 + 0.0040 * (1.0 - t))
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * warm_restart_phase))
    lr = floor + (peak - floor) * cosine

    late = jnp.clip((t - 0.66) / 0.34, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.91) / 0.09, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)

    coupled = 3.4 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    cycle_dip = 0.72 * (1.0 - 0.45 * t) * jnp.exp(-0.5 * (t_cycle / 0.16) ** 2)
    alpha = coupled * (1.0 - cycle_dip)
    alpha = alpha * (1.0 + 3.0 * t + 18.0 * late * late + 95.0 * tail * tail)

    beta1 = 0.10 + 0.10 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
