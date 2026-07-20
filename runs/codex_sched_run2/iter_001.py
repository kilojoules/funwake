"""Cosine LR with inverse penalty coupling.

HYPOTHESIS: A smooth cosine decay from full-size exploratory steps to a small
late step will preserve early wake-layout movement while letting constraints
settle without the sharper reciprocal decay shape in the seed.
AXIS: lr_cosine with TopFarm low-momentum Adam and alpha coupled inversely to lr.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    lr_floor = 0.002
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr_scale = lr_floor + (1.0 - lr_floor) * cosine
    lr = lr0 * lr_scale

    repair_ramp = 1.0 + 2.0 * t * t
    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10) * repair_ramp

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
