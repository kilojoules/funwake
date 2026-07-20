"""HYPOTHESIS: A smooth cosine LR taper preserves early wake-layout exploration while reducing late oscillation near active spacing and boundary constraints.
AXIS: lr_cosine with inverse-LR alpha coupling and TopFarm-low Adam betas.
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    hold_frac = 0.30
    min_lr_frac = 0.01

    phase = jnp.clip((t - hold_frac) / (1.0 - hold_frac), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * phase))
    lr_frac = min_lr_frac + (1.0 - min_lr_frac) * cosine
    lr = lr0 * jnp.where(t < hold_frac, 1.0, lr_frac)

    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
