"""HYPOTHESIS: The feasible linear_decay run gave up some AEP by lowering the
whole LR line; restoring a larger early slope while keeping the quiet terminal
step should recover layout quality, with a smoother alpha_linear_ramp handling
constraint repair.
AXIS: lr_linear_decay with alpha_linear_ramp plus mild quadratic tail.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = lr0 * (1.0 - t)
    lr = lr0 * (0.00025 + 4.24 * (lr / jnp.maximum(lr0, 1e-12)))

    early_relax = jnp.clip(1.0 - t / 0.17, 0.0, 1.0)
    late = jnp.clip((t - 0.50) / 0.50, 0.0, 1.0)
    tail = jnp.clip((t - 0.82) / 0.18, 0.0, 1.0)
    final = jnp.clip((t - 0.92) / 0.08, 0.0, 1.0)

    alpha = alpha0 * (1.0 + 9.2 * t + 52.0 * late * late
                      + 135.0 * tail * tail + 105.0 * final * final)
    alpha = alpha * (1.0 - 0.32 * early_relax)

    beta1 = 1.7e-1 + 1.0e-1 * t
    beta2 = 4.3e-1 + 1.8e-1 * t

    return lr, alpha, beta1, beta2
