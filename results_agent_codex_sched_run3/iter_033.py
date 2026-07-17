"""HYPOTHESIS: A schedule_two_phase shelf can preserve early basin search
better than decaying immediately, while the second phase uses the proven quiet
linear terminal repair from the best feasible linear-decay trial.
AXIS: schedule_two_phase with lr_linear_decay refinement and smooth alpha ramp.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    switch = 0.09
    phase = jnp.clip((t - switch) / jnp.maximum(1.0 - switch, 1e-12), 0.0, 1.0)
    decay_line = 0.00032 + 3.95 * (1.0 - phase)
    lr = lr0 * jnp.where(t < switch, 3.95, decay_line)

    early_relax = jnp.clip(1.0 - t / 0.20, 0.0, 1.0)
    late = jnp.clip((t - 0.50) / 0.50, 0.0, 1.0)
    tail = jnp.clip((t - 0.79) / 0.21, 0.0, 1.0)
    final = jnp.clip((t - 0.91) / 0.09, 0.0, 1.0)

    alpha = alpha0 * (2.10 + 8.0 * t + 42.0 * late * late
                      + 165.0 * tail * tail + 175.0 * final * final)
    alpha = alpha * (1.0 - 0.40 * early_relax)

    beta1 = 1.6e-1 + 1.1e-1 * t
    beta2 = 4.3e-1 + 1.7e-1 * t

    return lr, alpha, beta1, beta2
