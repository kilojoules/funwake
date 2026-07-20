"""HYPOTHESIS: The first linear_decay trial found a decent basin but missed
feasibility, so a lower terminal LR plus much stronger final alpha pressure
should repair constraints without changing the core straight-line LR shape.
AXIS: lr_linear_decay with stronger alpha_quadratic_ramp terminal repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0 * (1.0 - t)
    lr = lr0 * (0.0012 + 4.05 * (lr / jnp.maximum(lr0, 1e-12)))

    early_relax = jnp.clip(1.0 - t / 0.16, 0.0, 1.0)
    late = jnp.clip((t - 0.48) / 0.52, 0.0, 1.0)
    tail = jnp.clip((t - 0.80) / 0.20, 0.0, 1.0)
    final = jnp.clip((t - 0.93) / 0.07, 0.0, 1.0)

    alpha = alpha0 * (2.35 + 8.8 * t + 45.0 * late * late
                      + 360.0 * tail * tail + 1250.0 * final * final)
    alpha = alpha * (1.0 - 0.38 * early_relax)

    beta1 = 0.17 + 0.10 * t
    beta2 = 0.42 + 0.18 * t

    return lr, alpha, beta1, beta2
