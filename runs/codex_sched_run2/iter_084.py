"""Deterministic rational coast with a late penalty clutch.

HYPOTHESIS: The noisy exponential winner may be finding its basin because it
keeps enough mid-run mobility, not because the exponential tail is essential.
A rational coast preserves useful motion longer than exp(-8 t^2), then a
deliberate late quench and penalty clutch repair constraints without relying
on per-step random LR jitter.
AXIS: piecewise rational LR coast plus late quench, no stochastic noise.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    # Smooth high-mobility coast: starts near 4*lr0, matches the incumbent
    # around mid-run, and stays less quenched before the final repair phase.
    coast = lr0 * (0.09 + 3.91 / (1.0 + 34.0 * t * t))

    # Final quench begins after the layout has had time to separate wakes.
    q = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.72) / 0.28))
    q = q * q * (3.0 - 2.0 * q)
    tail = lr0 * (0.00125 + 0.18 * (1.0 - q) ** 3)
    lr = (1.0 - q) * coast + q * tail

    # Couple alpha to the quenched LR so the final low-step phase has enough
    # boundary pressure on harder polygons.
    alpha_base = 18.0 * alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    alpha = alpha_base * (1.0 + 0.28 * t + 0.10 * t * t)

    clutch = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.64) / 0.36))
    clutch = clutch * clutch * (3.0 - 2.0 * clutch)
    alpha = alpha * (1.0 + 8.0 * clutch + 68.0 * clutch * clutch)

    beta1 = 0.085 - 0.035 * clutch
    beta2 = 0.20 - 0.08 * clutch
    return lr, alpha, beta1, beta2
