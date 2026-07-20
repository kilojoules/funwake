"""Iter 197: Cosine warm restarts (SGDR-style) with 3 cycles.

Each cycle restarts LR to a high value and decays to near zero.
Successive cycles have decreasing peak LR (T_mult=1, lr_mult=0.7).
This helps escape local minima by periodically increasing step size.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0

    # 3 equal-length cycles
    n_cycles = 3
    cycle_len = 1.0 / n_cycles
    cycle_idx = jnp.floor(t / cycle_len)
    cycle_t = (t - cycle_idx * cycle_len) / cycle_len  # 0 to 1 within cycle

    # Each cycle peak decreases: lr_init * 0.7^cycle_idx
    cycle_peak = lr_init * (0.7 ** cycle_idx)
    lr_min = lr_init / 10000.0

    # Cosine decay within each cycle
    lr = lr_min + (cycle_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_t))

    # Alpha: coupled to lr, with late ramp
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
