"""HYPOTHESIS: A middle repair strength between iter_010 and iter_011 may keep
the SGDR objective gain while still clearing the held-out boundary check.
AXIS: lr_sgdr_warm_restarts with balanced late repair and shorter squeeze.
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
    floor = lr0 * (0.0016 + 0.0044 * (1.0 - t))
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * warm_restart_phase))
    lr = floor + (peak - floor) * cosine

    late = jnp.clip((t - 0.63) / 0.37, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.875) / 0.125, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.98) / 0.02, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    coupled = 3.4 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    cycle_dip = 0.71 * (1.0 - 0.45 * t) * jnp.exp(-0.5 * (t_cycle / 0.16) ** 2)
    alpha = coupled * (1.0 - cycle_dip)
    alpha = alpha * (1.0 + 3.0 * t + 21.0 * late * late + 160.0 * tail * tail)
    alpha = alpha * (1.0 + 65.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00115 - 0.00065 * squeeze), lr)

    beta1 = 0.10 + 0.10 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
