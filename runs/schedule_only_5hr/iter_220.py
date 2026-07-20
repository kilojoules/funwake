"""Iter 220: Polynomial warmup + decay with standard Adam betas.

Entirely different from cosine. Uses quadratic warmup (10%) then
inverse-square-root decay. Standard Adam betas (0.9, 0.999) give
strong momentum — different from the low-beta regime of all previous
attempts. Higher lr (6x) to compensate for Adam's normalization.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_peak = 6.0 * lr0
    lr_min = lr0 / 100.0
    warmup_end = 0.10

    # Quadratic warmup
    warmup_lr = lr_peak * (t / warmup_end) ** 2

    # Inverse square root decay after warmup
    decay_t = (t - warmup_end) / (1.0 - warmup_end)
    decay_lr = lr_peak / jnp.sqrt(1.0 + 20.0 * decay_t)
    decay_lr = jnp.maximum(decay_lr, lr_min)

    lr = jnp.where(t < warmup_end, warmup_lr, decay_lr)

    # Alpha coupled to 1/lr
    alpha = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
    alpha = jnp.minimum(alpha, 50.0 * alpha0)
    # Quadratic extra penalty from t=0.5
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha + 4.0 * alpha0 * late ** 2

    # Standard Adam betas
    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
