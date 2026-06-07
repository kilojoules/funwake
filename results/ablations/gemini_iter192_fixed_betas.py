"""Ablation: Gemini iter_192 with FIXED beta1=0.3, beta2=0.5.

Tests whether beta scheduling is necessary in Gemini's solution. lr/alpha
schedules (including final squeeze) are unchanged; only beta scheduling
is removed. Constants match the Claude iter_192 schedule.
"""
import jax
import jax.numpy as jnp


BETA1_FIXED = 0.3
BETA2_FIXED = 0.5


def schedule_fn(step, total_steps, lr0, alpha0):
    is_long_final = (step >= 4000)
    c_start = jnp.where(is_long_final, 4000.0, jnp.floor(step / 800.0) * 800.0)
    c_len = jnp.where(is_long_final, 4000.0, 800.0)

    t_cycle = (step - c_start) / (c_len - 1.0)
    t_global = step / (total_steps - 1.0)

    lr_peak = lr0 * 18.0 * (1.0 - 0.85 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))

    alpha_global_scale = 1.0 + 1200.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale

    dip_magnitude = 0.98 * (1.0 - 0.6 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle ** 2) / (2 * dip_width ** 2))

    alpha_local = alpha_global * (lr0 * 12.0 / jnp.maximum(lr, 1e-10))
    alpha = alpha_local * (1.0 - dip)

    is_squeeze = (t_global > 0.985)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e12, alpha)

    beta1 = jnp.full_like(lr, BETA1_FIXED)
    beta2 = jnp.full_like(lr, BETA2_FIXED)

    return lr, alpha, beta1, beta2
