"""Iter 315: Plateau 25% + bumps + high-alpha feasibility burst at t=0.90.

Same base as iter_298 but adds a final "feasibility burst":
briefly spike LR to 0.15*lr_init at t=0.90 with very high alpha
to push any remaining infeasible turbines into bounds.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    plateau_end = 0.25
    cosine_t = jnp.clip((t - plateau_end) / (1.0 - plateau_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < plateau_end, lr_init, cosine_lr)

    bump1 = 0.25 * lr_init * jnp.exp(-0.5 * ((t - 0.55) / 0.04) ** 2)
    bump2 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.78) / 0.03) ** 2)
    # Feasibility burst at 0.90
    bump3 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.90) / 0.02) ** 2)
    lr = lr_base + bump1 + bump2 + bump3

    # Alpha: inverse coupling + strong late ramp
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    # Extra spike during feasibility burst
    feas_burst = 20.0 * alpha0 * jnp.exp(-0.5 * ((t - 0.90) / 0.02) ** 2)
    alpha = alpha_base + alpha_extra + feas_burst

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
