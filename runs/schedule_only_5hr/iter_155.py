"""Iter 155: Cosine warm restarts (3 cycles) + transitioning betas.

Unlike all recent attempts (single cosine + bump), this uses SGDR-style
warm restarts: 3 cosine cycles with decreasing peak LR. At each restart,
alpha briefly dips to allow exploration before re-tightening.

Betas transition from high momentum (0.6) early to low momentum (0.15) late,
instead of fixed 0.3/0.5.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # --- Cosine warm restarts: 3 cycles (40%, 35%, 25% of budget) ---
    # Cycle boundaries
    c1_end = 0.40
    c2_end = 0.75
    # c3 runs to 1.0

    # Peak LR decreases each cycle
    peak1 = lr_init
    peak2 = 0.4 * lr_init
    peak3 = 0.1 * lr_init

    # Which cycle are we in?
    in_c1 = t < c1_end
    in_c2 = (t >= c1_end) & (t < c2_end)
    # in_c3 = t >= c2_end

    # Progress within each cycle (0 to 1)
    t1 = t / c1_end
    t2 = (t - c1_end) / (c2_end - c1_end)
    t3 = (t - c2_end) / (1.0 - c2_end)

    # Cosine annealing within each cycle
    lr_c1 = lr_min + (peak1 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t1))
    lr_c2 = lr_min + (peak2 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t2))
    lr_c3 = lr_min + (peak3 - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t3))

    lr = jnp.where(in_c1, lr_c1, jnp.where(in_c2, lr_c2, lr_c3))

    # --- Alpha: coupled to 1/lr but with brief dips at restarts ---
    # Base alpha coupled to lr
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # Dip alpha at restart points (Gaussian dips at c1_end and c2_end)
    dip1 = 0.5 * jnp.exp(-0.5 * ((t - c1_end) / 0.02) ** 2)
    dip2 = 0.5 * jnp.exp(-0.5 * ((t - c2_end) / 0.02) ** 2)
    alpha = alpha_base * (1.0 - dip1 - dip2)

    # --- Betas: transition from high to low momentum ---
    beta1 = 0.6 - 0.45 * t  # 0.6 -> 0.15
    beta2 = 0.7 - 0.5 * t   # 0.7 -> 0.2

    return lr, alpha, beta1, beta2
