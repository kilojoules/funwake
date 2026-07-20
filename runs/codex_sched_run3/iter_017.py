"""HYPOTHESIS: A front-loaded one-cycle schedule may recover more of the
restart-family objective gain by reaching its maximum step size earlier,
then spending most of the run in repair-aware annealing.
AXIS: lr_one_cycle with early peak and stronger inverse-LR alpha coupling.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    peak_pos = 0.12
    rise = 0.46 + 0.54 * t / peak_pos
    fall_base = jnp.clip((1.0 - t) / (1.0 - peak_pos), 0.0, 1.0)
    fall = fall_base * fall_base
    one_cycle = jnp.where(t < peak_pos, rise, fall)
    one_cycle = jnp.clip(one_cycle, 0.0, 1.0)

    peak = lr0 * (4.55 - 3.25 * t)
    floor = lr0 * (0.00135 + 0.0078 * (1.0 - t))
    lr = floor + (peak - floor) * one_cycle

    repair = jnp.clip((t - 0.60) / 0.40, 0.0, 1.0)
    repair = repair * repair * (3.0 - 2.0 * repair)
    tail = jnp.clip((t - 0.865) / 0.135, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    coupled = 3.12 * alpha0 * lr0 / jnp.maximum(lr + 0.045 * lr0, 1e-10)
    alpha_dip = 0.60 * (1.0 - 0.46 * t) * one_cycle
    alpha = coupled * (1.0 - alpha_dip)
    alpha = alpha * (1.0 + 2.8 * t + 20.0 * repair * repair + 145.0 * tail * tail)
    alpha = alpha * (1.0 + 56.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00106 - 0.00065 * squeeze), lr)

    beta1 = 0.09 + 0.105 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
