"""Self-contained scale-aware seed: cosine decay + two exploratory lr bumps and
coupled+ramped alpha (iter192-like). peak = 0.8333*D, cosine to lr_min, with
gaussian re-exploration bumps at t=0.5 and t=0.75. Descriptor cell: peak_lr/D in
[0.8,1.2], coupled, >=2 restarts (distinct restart bin from the monotone native
seed).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    t = step / total_steps
    lr_init = 0.8333 * D
    lr_min = jnp.maximum(gamma_min, lr_init / 10000.0)
    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    bump1 = 0.20 * lr_init * jnp.exp(-0.5 * ((t - 0.50) / 0.04) ** 2)
    bump2 = 0.15 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump1 + bump2
    alpha_base = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha_base + 3.0 * alpha0 * late ** 2
    beta1 = 0.3
    beta2 = 0.5
    return lr, alpha, beta1, beta2
