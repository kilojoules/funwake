"""Iter 174: Warm restarts with decaying amplitude + longer final phase.

Restarts go to 75%, 50% of initial LR (not full reset).
Final phase gets 40% of budget for thorough refinement.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # 3 cycles: 30%, 30%, 40%
    c1_end = 0.30
    c2_end = 0.60

    in_c1 = t < c1_end
    in_c2 = (t >= c1_end) & (t < c2_end)

    t_local = jnp.where(
        in_c1,
        t / c1_end,
        jnp.where(
            in_c2,
            (t - c1_end) / (c2_end - c1_end),
            (t - c2_end) / (1.0 - c2_end)
        )
    )

    # Decaying restart amplitude: 100%, 75%, 50% of lr_init
    lr_max = jnp.where(in_c1, lr_init, jnp.where(in_c2, 0.75 * lr_init, 0.5 * lr_init))
    lr = lr_min + (lr_max - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t_local))

    # Alpha: global trend increasing + cycle coupling
    global_alpha = 3.0 + 7.0 * t  # 3x -> 10x over full run
    alpha = global_alpha * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Beta1 ramps down smoothly over whole run
    beta1 = 0.5 - 0.35 * t  # 0.5 -> 0.15
    beta2 = 0.5

    return lr, alpha, beta1, beta2
