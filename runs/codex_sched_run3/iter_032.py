"""HYPOTHESIS: Linear decay performed best when it was quieter early and firm
late; a slightly lower initial line with a smoother but stronger tail alpha
may improve the feasible iter_030 basin without reviving abrupt repair chatter.
AXIS: lr_linear_decay with alpha_quadratic_ramp and quiet terminal repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0 * (1.0 - t)
    lr = lr0 * (0.00022 + 3.82 * (lr / jnp.maximum(lr0, 1e-12)))

    early_relax = jnp.clip(1.0 - t / 0.18, 0.0, 1.0)
    late = jnp.clip((t - 0.46) / 0.54, 0.0, 1.0)
    tail = jnp.clip((t - 0.76) / 0.24, 0.0, 1.0)
    final = jnp.clip((t - 0.90) / 0.10, 0.0, 1.0)

    alpha = alpha0 * (2.30 + 8.6 * t + 50.0 * late * late
                      + 190.0 * tail * tail + 230.0 * final * final)
    alpha = alpha * (1.0 - 0.34 * early_relax)

    beta1 = 1.6e-1 + 1.0e-1 * t
    beta2 = 4.4e-1 + 1.7e-1 * t

    return lr, alpha, beta1, beta2
