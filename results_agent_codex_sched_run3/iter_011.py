"""HYPOTHESIS: The first SGDR schedule found a better objective basin but left
held-out boundary drift, so an earlier tail repair and final constraint squeeze
should preserve most of the gain while improving generalization feasibility.
AXIS: lr_sgdr_warm_restarts with stronger late boundary/spacing repair.
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
    floor = lr0 * (0.0014 + 0.0046 * (1.0 - t))
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * warm_restart_phase))
    lr = floor + (peak - floor) * cosine

    late = jnp.clip((t - 0.61) / 0.39, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.86) / 0.14, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.975) / 0.025, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    coupled = 3.4 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    cycle_dip = 0.70 * (1.0 - 0.45 * t) * jnp.exp(-0.5 * (t_cycle / 0.16) ** 2)
    alpha = coupled * (1.0 - cycle_dip)
    alpha = alpha * (1.0 + 3.0 * t + 24.0 * late * late + 210.0 * tail * tail)
    alpha = alpha * (1.0 + 100.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.0010 - 0.00065 * squeeze), lr)

    beta1 = 0.10 + 0.10 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
