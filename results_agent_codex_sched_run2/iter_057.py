"""Baseline-calibrated dual-bump cosine schedule.

HYPOTHESIS: The noisy exponential schedules appear capped near 5564 GWh in
this run. A distinct warmup-cosine backbone with two learned LR bumps and
moderate Adam memory may reach a different layout basin while keeping the
strong feasibility behavior seen in the bump baseline.
AXIS: warmup cosine plus dual Gaussian LR bumps and moderate Adam betas.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    k = 5.526989
    log_m = 3.858923
    warmup = 0.054557
    bump1_amp = 0.152736
    bump2_amp = 0.408010
    bump1_center = 0.821488
    bump2_center = 0.391711
    bump1_width = 0.107955
    bump2_width = 0.079159
    alpha_coupling = 3.968751
    alpha_late = 11.756615
    beta1 = 0.221956
    beta2 = 0.509058

    lr_init = k * lr0
    lr_min = lr_init / (10.0 ** log_m)

    warmup_lr = lr_init * t / jnp.maximum(warmup, 1e-6)
    cosine_t = (t - warmup) / jnp.maximum(1.0 - warmup, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (
        1.0 + jnp.cos(jnp.pi * cosine_t)
    )
    lr_base = jnp.where(t < warmup, warmup_lr, cosine_lr)

    bump1 = bump1_amp * lr_init * jnp.exp(
        -0.5 * ((t - bump1_center) / bump1_width) ** 2
    )
    bump2 = bump2_amp * lr_init * jnp.exp(
        -0.5 * ((t - bump2_center) / bump2_width) ** 2
    )
    lr = jnp.maximum(lr_base + bump1 + bump2, 1e-10)

    alpha_base = alpha_coupling * alpha0 * lr_init / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha_base + alpha_late * alpha0 * late ** 2

    return lr, alpha, beta1, beta2
