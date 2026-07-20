"""One-cycle super-convergence: ramp LR above lr0 then decay.

The high-LR phase acts as implicit regularization, helping the optimizer
explore more broadly from a single start. Combined with reduced alpha
during exploration to prioritize AEP over constraints.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    # Phases: ramp-up (25%), ramp-down (45%), enforcement (30%)
    peak_time = 0.25
    decay_end = 0.70

    # Peak LR is 2x lr0 for broader exploration
    lr_peak = 2.0 * lr0

    # === Learning rate ===
    # Ramp up: lr0/5 to 2*lr0
    ramp_progress = t / peak_time
    lr_ramp = lr0 * 0.2 + (lr_peak - lr0 * 0.2) * ramp_progress

    # Ramp down: cosine from 2*lr0 to 0.02*lr0
    decay_progress = (t - peak_time) / (decay_end - peak_time)
    lr_decay = lr0 * (0.02 + (lr_peak/lr0 - 0.02) * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress)))

    # Enforcement: exponential to very low
    enforce_progress = (t - decay_end) / (1.0 - decay_end)
    lr_enforce = lr0 * 0.02 * jnp.exp(-5.0 * enforce_progress)

    lr = jnp.where(t < peak_time, lr_ramp,
         jnp.where(t < decay_end, lr_decay, lr_enforce))

    # === Alpha: reduced during ramp-up to prioritize AEP ===
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    multiplier = jnp.where(t < peak_time, 0.5,
                 jnp.where(t >= decay_end, 100.0, 1.0))
    alpha = base_alpha * multiplier

    # === Momentum ===
    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
