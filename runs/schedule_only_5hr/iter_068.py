"""Iter 068: Two-phase Adam betas + cosine LR decay.

Phase 1 (0-35%): High LR, standard Adam (beta1=0.9, beta2=0.999) for exploration.
Phase 2 (35-100%): Cosine LR decay, low momentum (0.1, 0.2) for fine-tuning.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 3.0 * lr0
    lr_min = lr_init / 30000.0
    phase_switch = 0.35

    in_phase1 = t < phase_switch

    # Phase 2: cosine decay
    decay_t = jnp.maximum(t - phase_switch, 0.0) / (1.0 - phase_switch)
    lr_cos = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * decay_t))
    lr = jnp.where(in_phase1, lr_init, lr_cos)

    # Alpha coupled to 1/lr
    alpha = jnp.where(in_phase1, 3.0 * alpha0,
                      alpha0 * lr_init / jnp.maximum(lr, 1e-10))

    # Two-phase betas
    beta1 = jnp.where(in_phase1, 0.9, 0.1)
    beta2 = jnp.where(in_phase1, 0.999, 0.2)
    return lr, alpha, beta1, beta2
