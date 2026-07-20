"""Iter 084: Full cosine, 4x LR, lr/5000, extra alpha boost in last 20%."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 5000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # Standard alpha coupling, but boost in last 20% for feasibility
    base_alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late_boost = jnp.where(t > 0.8, 3.0, 1.0)
    alpha = base_alpha * late_boost

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
