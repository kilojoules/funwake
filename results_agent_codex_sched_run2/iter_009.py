"""Linear LR decay with capped inverse penalty coupling.

HYPOTHESIS: A monotone linear learning-rate decay can preserve useful early
layout motion while avoiding the disruptive pulse/restart behavior that hurt
several previous attempts.
AXIS: lr_linear_decay plus alpha_coupled_inverse_lr, TopFarm low-momentum Adam,
and a mild final alpha repair cap.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    hold = 0.20
    decay_t = jnp.maximum(0.0, (t - hold) / (1.0 - hold))
    terminal = 0.006

    lr_frac = 1.0 - (1.0 - terminal) * decay_t
    lr = lr0 * lr_frac

    alpha_coupled = alpha0 / jnp.maximum(lr_frac, 1e-8)

    repair_t = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.72) / 0.28))
    repair = repair_t * repair_t * (3.0 - 2.0 * repair_t)
    alpha = alpha_coupled * (1.0 + 8.0 * repair)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
