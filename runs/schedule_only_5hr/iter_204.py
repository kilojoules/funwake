"""Iter 204: Piecewise linear LR with 4 breakpoints.

Hand-tuned breakpoints instead of smooth functions.
Phase 1 (0-5%): warmup 0->peak
Phase 2 (5-30%): hold at peak (exploration)
Phase 3 (30-80%): linear decay to 10% of peak
Phase 4 (80-100%): slow decay to minimum (refinement)
No bump needed — the flat exploration phase serves same purpose.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_peak = 4.0 * lr0
    lr_min = lr_peak / 10000.0

    # Piecewise linear: warmup -> flat -> decay -> fine-tune
    lr = jnp.where(t < 0.05,
            lr_peak * t / 0.05,
         jnp.where(t < 0.30,
            lr_peak,
         jnp.where(t < 0.80,
            lr_peak - (lr_peak - 0.1 * lr_peak) * (t - 0.30) / 0.50,
            0.1 * lr_peak - (0.1 * lr_peak - lr_min) * (t - 0.80) / 0.20)))

    alpha_base = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
