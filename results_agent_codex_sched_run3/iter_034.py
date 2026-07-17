"""HYPOTHESIS: The first schedule_two_phase shelf was too aggressive for thin
polygons; a shorter, lower shelf should keep the qualitative explore/refine
switch while leaving enough quiet terminal motion for boundary repair.
AXIS: conservative schedule_two_phase with lr_linear_decay refinement.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    switch = 0.045
    phase = jnp.clip((t - switch) / jnp.maximum(1.0 - switch, 1e-12), 0.0, 1.0)
    decay_line = 0.00025 + 3.72 * (1.0 - phase)
    lr = lr0 * jnp.where(t < switch, 3.72, decay_line)

    early_relax = jnp.clip(1.0 - t / 0.16, 0.0, 1.0)
    late = jnp.clip((t - 0.47) / 0.53, 0.0, 1.0)
    tail = jnp.clip((t - 0.76) / 0.24, 0.0, 1.0)
    final = jnp.clip((t - 0.90) / 0.10, 0.0, 1.0)

    alpha = alpha0 * (2.45 + 8.8 * t + 54.0 * late * late
                      + 210.0 * tail * tail + 230.0 * final * final)
    alpha = alpha * (1.0 - 0.30 * early_relax)

    beta1 = 1.55e-1 + 1.05e-1 * t
    beta2 = 4.4e-1 + 1.7e-1 * t

    return lr, alpha, beta1, beta2
