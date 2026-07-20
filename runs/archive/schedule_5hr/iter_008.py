"""Standard Adam momentum (0.9/0.999) with TopFarm-style decay.

Hypothesis: standard Adam momentum is better at navigating the AEP
landscape than TopFarm's low momentum (0.1/0.2).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    const_end = 0.60
    decay_end = 0.80

    # Standard TopFarm-style decay schedule
    lr_const = lr0

    decay_steps = total_steps * (decay_end - const_end)
    mid = 99.0 / jnp.maximum(decay_steps, 1.0)
    decay_step = (t - const_end) * total_steps
    lr_decay = lr0 / (1 + mid * jnp.maximum(decay_step, 0.0))

    enforce_progress = (t - decay_end) / (1.0 - decay_end)
    lr_enforce = lr0 * 0.01 * jnp.exp(-3.0 * enforce_progress)

    lr = jnp.where(t < const_end, lr_const,
         jnp.where(t < decay_end, lr_decay, lr_enforce))

    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    boost = jnp.where(t >= decay_end, 50.0, 1.0)
    alpha = jnp.where(t < const_end, alpha0, base_alpha * boost)

    # Standard Adam momentum
    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
