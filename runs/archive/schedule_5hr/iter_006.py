"""Extended constant-LR: 75% constant, 15% decay, 10% enforcement.

Hypothesis: more constant-LR steps = more time for the single start
to explore and find a good basin.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    const_end = 0.75
    decay_end = 0.90

    # Constant phase
    lr_const = lr0

    # Decay: 1/(1+mid*t) style targeting 0.01*lr0
    decay_progress = (t - const_end) / (decay_end - const_end)
    decay_steps_equiv = total_steps * (decay_end - const_end)
    mid = 99.0 / jnp.maximum(decay_steps_equiv, 1.0)
    decay_step = (t - const_end) * total_steps
    lr_decay = lr0 / (1 + mid * decay_step)

    # Enforcement: continue decay with alpha boost
    enforce_progress = (t - decay_end) / (1.0 - decay_end)
    lr_enforce = lr0 * 0.01 * jnp.exp(-3.0 * enforce_progress)

    lr = jnp.where(t < const_end, lr_const,
         jnp.where(t < decay_end, lr_decay, lr_enforce))

    # Alpha: coupled + boost in enforcement
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    boost = jnp.where(t >= decay_end, 50.0, 1.0)
    # During constant phase, keep alpha at alpha0
    alpha = jnp.where(t < const_end, alpha0, base_alpha * boost)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
