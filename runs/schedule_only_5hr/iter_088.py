"""Iter 088: Full cosine 90% + feasibility phase 10%, 4x LR, lr/5000."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 5000.0
    feas_start = 0.90

    in_opt = t < feas_start

    # Optimization phase: cosine decay over 90% of steps
    opt_t = t / feas_start
    lr_cos = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * opt_t))

    # Feasibility phase: very low LR, very high alpha
    lr_feas = lr_min * 0.1

    lr = jnp.where(in_opt, lr_cos, lr_feas)
    alpha_opt = alpha0 * lr_init / jnp.maximum(lr_cos, 1e-10)
    alpha_feas = alpha0 * lr_init / lr_feas
    alpha = jnp.where(in_opt, alpha_opt, alpha_feas)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
