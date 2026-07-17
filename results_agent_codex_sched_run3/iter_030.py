"""HYPOTHESIS: Linear decay needs feasibility from a quiet terminal step more
than from a very large alpha spike; a low final LR with a smoother quadratic
alpha ramp should preserve the iter_028 basin while avoiding boundary chatter.
AXIS: lr_linear_decay with smooth alpha_quadratic_ramp terminal repair.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0 * (1.0 - t)
    lr = lr0 * (0.00035 + 3.95 * (lr / jnp.maximum(lr0, 1e-12)))

    early_relax = jnp.clip(1.0 - t / 0.18, 0.0, 1.0)
    late = jnp.clip((t - 0.48) / 0.52, 0.0, 1.0)
    tail = jnp.clip((t - 0.78) / 0.22, 0.0, 1.0)
    final = jnp.clip((t - 0.90) / 0.10, 0.0, 1.0)

    alpha = alpha0 * (2.25 + 8.2 * t + 42.0 * late * late
                      + 170.0 * tail * tail + 180.0 * final * final)
    alpha = alpha * (1.0 - 0.36 * early_relax)

    beta1 = 0.16 + 0.11 * t
    beta2 = 0.43 + 0.17 * t

    return lr, alpha, beta1, beta2
