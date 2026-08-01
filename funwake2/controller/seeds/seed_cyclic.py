"""Self-contained scale-aware seed: cyclic warm-restart lr with cyclic alpha
(iter118-like). Hotter peak (~1.35*D) with several cosine cycles and per-cycle
alpha dips. Descriptor cell: peak_lr/D > 1.2, cyclic coupling, >=3 restarts.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    t = step / total_steps
    n_cycles = 6.0
    cyc = (step % (total_steps / n_cycles)) / (total_steps / n_cycles)
    lr_peak = 1.35 * D * (1.0 - 0.5 * t)
    lr_min = jnp.maximum(gamma_min, 0.005 * D)
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * cyc))
    alpha_coupled = alpha0 * D * (1.0 + 100.0 * t ** 3) / jnp.maximum(lr, 1e-10)
    dip = 0.9 * jnp.exp(-(cyc ** 2) / (2 * 0.12 ** 2))
    alpha = alpha_coupled * (1.0 - dip)
    beta1 = 0.1 + 0.3 * cyc
    beta2 = 0.2 + 0.7 * cyc
    return lr, alpha, beta1, beta2
