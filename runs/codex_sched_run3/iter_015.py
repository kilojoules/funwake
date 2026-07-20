import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    cycle_len = total_steps / 9.0
    final_start = 5.0 * cycle_len
    in_final = step >= final_start

    cycle_start = jnp.where(in_final, final_start,
                            jnp.floor(step / cycle_len) * cycle_len)
    active_len = jnp.where(in_final, total_steps - final_start, cycle_len)
    p = (step - cycle_start) / jnp.maximum(active_len - 1.0, 1.0)

    peak_pos = 0.18
    rise = 0.58 + 0.42 * p / peak_pos
    fall = (1.0 - p) / (1.0 - peak_pos)
    tri = jnp.where(p < peak_pos, rise, fall)
    tri = jnp.clip(tri, 0.0, 1.0)

    peak = lr0 * (4.95 - 3.55 * t)
    floor = lr0 * (0.00135 + 0.0062 * (1.0 - t))
    lr = floor + (peak - floor) * tri

    late = jnp.clip((t - 0.61) / 0.39, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    tail = jnp.clip((t - 0.86) / 0.14, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    coupled = 3.35 * alpha0 * lr0 / jnp.maximum(lr + 0.033 * lr0, 1e-10)
    peak_dip = 0.68 * (1.0 - 0.45 * t) * tri
    alpha = coupled * (1.0 - peak_dip)
    alpha = alpha * (1.0 + 3.0 * t + 24.0 * late * late + 190.0 * tail * tail)
    alpha = alpha * (1.0 + 80.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00105 - 0.00066 * squeeze), lr)

    beta1 = 0.09 + 0.11 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
