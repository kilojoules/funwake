"""Iter 199: Two-phase strategy — long exploration then tight refinement.

Phase 1 (0-60%): High LR, LOW alpha (constraints relaxed), high momentum.
  Turbines can move freely to find globally good positions.
Phase 2 (60-100%): Rapid LR decay, VERY high alpha, low momentum.
  Lock turbines into feasible positions with fine adjustments.
Transition is smooth (sigmoid blending).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0  # Even higher initial LR for exploration
    lr_min = lr_init / 10000.0

    # Sigmoid transition centered at t=0.6
    phase = 1.0 / (1.0 + jnp.exp(-30.0 * (t - 0.6)))  # 0 = explore, 1 = refine

    # LR: high and flat during exploration, rapid decay in refinement
    lr_explore = lr_init
    lr_refine = lr_min + (0.3 * lr_init - lr_min) * jnp.exp(-8.0 * (t - 0.6))
    lr = (1.0 - phase) * lr_explore + phase * lr_refine

    # Alpha: low during exploration, very high during refinement
    alpha_explore = 0.5 * alpha0
    alpha_refine = 10.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = (1.0 - phase) * alpha_explore + phase * alpha_refine

    # Beta: high momentum for exploration, low for refinement
    beta1 = (1.0 - phase) * 0.7 + phase * 0.1
    beta2 = (1.0 - phase) * 0.9 + phase * 0.3

    return lr, alpha, beta1, beta2
