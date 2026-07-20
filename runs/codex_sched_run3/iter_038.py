"""HYPOTHESIS: A noisy exponential cooling schedule with a tiny late mobility
shelf can recover part of the high-AEP infeasible run while keeping the
penalty envelope intact enough to finish feasible.
AXIS: lr_noise_injection with exponential cooling and tiny late LR shelf.
LESSON: Pending score.
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00038 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    random = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.clip((t - 0.12) / 0.18, 0.0, 1.0)
    fall = jnp.clip((0.72 - t) / 0.22, 0.0, 1.0)
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)
    lr = lr_base + lr_base * (0.08 * noise_gate) * random

    shelf = jnp.exp(-0.5 * ((t - 0.770) / 0.040) ** 2)
    shelf_start = jnp.clip((t - 0.710) / 0.040, 0.0, 1.0)
    shelf_start = shelf_start * shelf_start * (3.0 - 2.0 * shelf_start)
    shelf_stop = jnp.clip((0.835 - t) / 0.040, 0.0, 1.0)
    shelf_stop = shelf_stop * shelf_stop * (3.0 - 2.0 * shelf_stop)
    shelf = shelf * shelf_start * shelf_stop
    lr = jnp.maximum(lr, lr0 * (0.00042 + 0.0045 * shelf))
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)

    late = jnp.clip((t - 0.68) / 0.32, 0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
