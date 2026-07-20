"""Anti-phase alpha dip during the LR perturbation window.

HYPOTHESIS: The best feasible noisy schedule may still pay too much penalty
during the mid-run basin-selection window; dipping alpha only while LR noise is
active can preserve objective exploration, then a stronger late ramp can repair
the resulting boundary slack.
AXIS: alpha_anti_phase_dip paired with mid-course LR perturbation.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00032 * lr0
    exp_decay = jnp.exp(-8.00 * t * t)
    lr_base = jnp.maximum(lr_start * exp_decay, lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base + lr_base * (0.08 * noise_gate) * lr_noise
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)
    alpha = alpha * (1.0 - 0.12 * noise_gate)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.67) / 0.33))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 8.2 * late + 64.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
