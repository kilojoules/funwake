"""Iter 292: Two-phase — aggressive exploration then strict convergence.

Phase 1 (0-50%): High LR, low alpha, standard Adam betas → maximize AEP
Phase 2 (50-100%): Exponential LR decay, aggressive alpha ramp → enforce feasibility
Transition is smooth via sigmoid blending.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 5.0 * lr0
    lr_final = 0.01 * lr0

    # Sigmoid transition at t=0.5, steepness=20
    blend = 1.0 / (1.0 + jnp.exp(-20.0 * (t - 0.5)))

    # Phase 1: constant high LR
    lr_explore = lr_init

    # Phase 2: exponential decay from lr_init to lr_final
    decay_t = jnp.clip((t - 0.5) / 0.5, 0.0, 1.0)
    lr_converge = lr_init * jnp.exp(decay_t * jnp.log(lr_final / lr_init))

    lr = (1.0 - blend) * lr_explore + blend * lr_converge

    # Alpha: low in phase 1, ramps quadratically in phase 2
    alpha_base = alpha0
    alpha_late = alpha0 * (lr_init / jnp.maximum(lr, 1e-10))
    alpha = (1.0 - blend) * alpha_base + blend * alpha_late

    # Standard Adam early, TopFarm-like late
    beta1 = (1.0 - blend) * 0.9 + blend * 0.1
    beta2 = (1.0 - blend) * 0.999 + blend * 0.2

    return lr, alpha, beta1, beta2
