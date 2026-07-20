"""Two-phase basin search followed by hard constraint repair.

HYPOTHESIS: A clear phase boundary can preserve the useful large early moves
seen in high-start schedules, then hand the final third of the run to small
steps and very strong penalties instead of slowly blending the objectives.
AXIS: schedule_two_phase with high-LR exploration and low-LR penalty repair.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    phase = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.58) / 0.16))
    phase = phase * phase * (3.0 - 2.0 * phase)
    repair = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.60) / 0.40))
    repair = repair * repair * (3.0 - 2.0 * repair)

    lr_explore = 3.7 * lr0
    lr_refine = 0.004 * lr0
    lr = lr_explore * (1.0 - phase) + lr_refine * phase

    alpha_explore = alpha0 * (5.0 + 20.0 * t * t)
    alpha_refine = alpha0 * (55.0 + 300000.0 * repair * repair)
    alpha = alpha_explore * (1.0 - phase) + alpha_refine * phase

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
