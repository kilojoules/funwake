"""HYPOTHESIS: A true one-cycle LR can reach the high-AEP basin found by
restart schedules without repeatedly disrupting late constraint repair.
AXIS: lr_one_cycle with inverse-LR alpha recovery and a short final squeeze.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    up_end = 0.18
    anneal_end = 0.84

    up = jnp.clip(t / up_end, 0.0, 1.0)
    up = up * up * (3.0 - 2.0 * up)

    down = jnp.clip((t - up_end) / (anneal_end - up_end), 0.0, 1.0)
    down = down * down * (3.0 - 2.0 * down)

    start = lr0 * 0.72
    peak = lr0 * 5.15
    floor = lr0 * 0.00125
    warm_lr = start + (peak - start) * up
    cool_lr = floor + (peak - floor) * (1.0 - down) ** 2.15
    lr = jnp.where(t < up_end, warm_lr, cool_lr)

    repair = jnp.clip((t - 0.62) / 0.38, 0.0, 1.0)
    repair = repair * repair * (3.0 - 2.0 * repair)
    tail = jnp.clip((t - 0.875) / 0.125, 0.0, 1.0)
    tail = tail * tail * (3.0 - 2.0 * tail)
    squeeze = jnp.clip((t - 0.978) / 0.022, 0.0, 1.0)
    squeeze = squeeze * squeeze * (3.0 - 2.0 * squeeze)

    one_cycle_strength = up * (1.0 - down)
    coupled = 3.18 * alpha0 * lr0 / jnp.maximum(lr + 0.041 * lr0, 1e-10)
    alpha_dip = 0.64 * (1.0 - 0.48 * t) * one_cycle_strength
    alpha = coupled * (1.0 - alpha_dip)
    alpha = alpha * (1.0 + 2.7 * t + 22.0 * repair * repair + 165.0 * tail * tail)
    alpha = alpha * (1.0 + 68.0 * squeeze * squeeze)

    lr = jnp.where(squeeze > 0.0, lr0 * (0.00108 - 0.00064 * squeeze), lr)

    beta1 = 0.085 + 0.105 * t
    beta2 = 0.30 + 0.25 * t

    return lr, alpha, beta1, beta2
