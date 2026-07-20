"""Fixed scaled LR with quadratic feasibility pressure.

HYPOTHESIS: A fixed but smaller Adam step may keep wake-basin movement active
for the full run, while a convex penalty ramp supplies firmer spacing and
boundary repair without relying on LR cooling.
AXIS: lr_constant plus alpha_quadratic_ramp with low Adam smoothing.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0
    lr = lr * 0.08

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.58) / 0.42))
    late = late * late * (3.0 - 2.0 * late)

    alpha = alpha0 * (1.0 + 820.0 * t * t)
    alpha = 24.0 * alpha * (1.0 + 160.0 * late * late)

    beta1 = 1.0 / 10.0
    beta2 = 1.0 / 5.0

    return lr, alpha, beta1, beta2
