"""HYPOTHESIS: With the best local alpha counter-ramp in place, the mobility
kick may work better if LR peaks slightly before alpha so wake motion starts
under lower pressure and then tightens through the settling tail.
AXIS: lr_alpha_phase_split
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr_bump = jnp.exp(-0.5 * ((t - 0.688) / 0.05) ** 2)
    alpha_bump = jnp.exp(-0.5 * ((t - 0.699) / 0.047) ** 2)
    lr = lr_base + 0.3 * lr_init * lr_bump

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * late**2
    alpha = alpha + 1.18 * alpha0 * alpha_bump

    return lr, alpha, 0.3, 0.5
