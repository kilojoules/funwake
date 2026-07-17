"""HYPOTHESIS: A high-start linear_decay schedule may keep the useful early
basin-changing motion without the previous cosine/bump machinery, while a
late-heavy quadratic alpha ramp repairs polygon/spacing violations as the
step size shrinks.
AXIS: lr_linear_decay with alpha_quadratic_ramp and medium-low Adam momentum.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    # Linear decay from an aggressive exploratory scale to a small terminal step.
    lr = lr0 * (1.0 - t)
    lr = lr0 * (0.0045 + 4.15 * (lr / jnp.maximum(lr0, 1e-12)))

    early_relax = jnp.clip(1.0 - t / 0.18, 0.0, 1.0)
    late = jnp.clip((t - 0.52) / 0.48, 0.0, 1.0)
    tail = jnp.clip((t - 0.84) / 0.16, 0.0, 1.0)

    alpha = alpha0 * (2.15 + 7.6 * t + 31.0 * late * late + 95.0 * tail * tail)
    alpha = alpha * (1.0 - 0.42 * early_relax)

    beta1 = 0.18 + 0.10 * t
    beta2 = 0.44 + 0.16 * t

    return lr, alpha, beta1, beta2
