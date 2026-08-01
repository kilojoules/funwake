"""Self-contained scale-aware seed: native-like monotone decay to gamma_min.

Standalone (imports only jax.numpy) so the cascade can exec it from source and
the mutation engine can perturb its float literals. Exploration lr is built from
D (no free lr0): peak = 0.8333*D, constant for the first third, then geometric
decay to the absolute gamma_min. alpha is coupled (alpha0*D/lr). Descriptor cell:
peak_lr/D in [0.8,1.2], terminal ~gamma_min, coupled, 0 restarts.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    lr0 = 0.8333 * D
    n_const = total_steps // 3
    decay_len = jnp.maximum(total_steps - n_const, 1)
    frac = jnp.clip((step - n_const) / decay_len, 0.0, 1.0)
    ratio = jnp.maximum(gamma_min, 1e-6) / lr0
    lr = lr0 * jnp.power(ratio, frac)
    alpha = alpha0 * D / jnp.maximum(lr, 1e-30)
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
