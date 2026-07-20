"""One-cycle super-convergence + standard Adam momentum.

Combines the two best strategies:
- iter_007's one-cycle LR schedule (2x peak) that beat baseline
- iter_008's standard Adam momentum (0.9/0.999) that boosted AEP by 6 GWh
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    # Phases: ramp-up (25%), ramp-down (45%), enforcement (30%)
    peak_time = 0.25
    decay_end = 0.70

    lr_peak = 2.0 * lr0

    # === Learning rate (from iter_007) ===
    ramp_progress = t / peak_time
    lr_ramp = lr0 * 0.2 + (lr_peak - lr0 * 0.2) * ramp_progress

    decay_progress = (t - peak_time) / (decay_end - peak_time)
    lr_decay = lr0 * (0.02 + (lr_peak/lr0 - 0.02) * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress)))

    enforce_progress = (t - decay_end) / (1.0 - decay_end)
    lr_enforce = lr0 * 0.02 * jnp.exp(-5.0 * enforce_progress)

    lr = jnp.where(t < peak_time, lr_ramp,
         jnp.where(t < decay_end, lr_decay, lr_enforce))

    # === Alpha ===
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    multiplier = jnp.where(t < peak_time, 0.5,
                 jnp.where(t >= decay_end, 100.0, 1.0))
    alpha = base_alpha * multiplier

    # === Standard Adam momentum (from iter_008) ===
    beta1 = 0.9
    beta2 = 0.999

    return lr, alpha, beta1, beta2
