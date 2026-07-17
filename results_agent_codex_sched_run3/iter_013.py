import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    cycle_len = total_steps / 8.0
    final_start = 5.0 * cycle_len
    in_final = step >= final_start

    cycle_start = jnp.where(in_final, final_start,
                            jnp.floor(step / cycle_len) * cycle_len)
    active_len = jnp.where(in_final, total_steps - final_start, cycle_len)
    p = (step - cycle_start) / jnp.maximum(active_len - 1.0, 1.0)

    tri = 1.0 - jnp.abs(2.0 * p - 1.0)
    tri = jnp.clip(tri, 0.0, 1.0)

    peak = lr0 * (4.4 - 3.1 * t)
    floor = lr0 * (0.0015 + 0.0120 * (1.0 - t))
    lr = floor + (peak - floor) * tri

    refine = jnp.clip((t - 0.76) / 0.24, 0.0, 1.0)
    refine = refine * refine * (3.0 - 2.0 * refine)
    tail = jnp.clip((t - 0.89) / 0.11, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.977) / 0.023, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    alpha_coupled = 3.0 * alpha0 * lr0 / jnp.maximum(lr + 0.055 * lr0, 1e-10)
    high_lr_dip = 0.55 * (1.0 - 0.55 * t) * tri
    alpha = alpha_coupled * (1.0 - high_lr_dip)
    alpha = alpha * (1.0 + 2.5 * t + 18.0 * refine * refine + 125.0 * tail * tail)
    alpha = alpha * (1.0 + 55.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00110 - 0.00062 * squeeze), lr)

    beta1 = 0.08 + 0.10 * t
    beta2 = 0.28 + 0.24 * t

    return lr, alpha, beta1, beta2
