"""HYPOTHESIS: A brief anti-phase penalty dip during a mobility pulse can let the wake layout escape early local arrangements, while a late penalty recovery preserves boundary and spacing feasibility.
AXIS: alpha_anti_phase_dip
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    # Normalize to [0, 1] without dividing by zero on tiny test loops.
    denom = jnp.maximum(total_steps - 1, 1)
    t = step / denom

    const_phase = total_steps // 3
    decay_steps = jnp.maximum(total_steps - const_phase, 1.0)
    decay_step = jnp.maximum(step - const_phase, 0.0)
    decaying = step >= const_phase

    # TopFarm-like base: fixed early LR, then rational decay to 1% lr0.
    mid = 99.0 / decay_steps
    base_lr = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)
    base_alpha = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(base_lr, 1e-10),
        alpha0,
    )

    # Anti-phase event: increase mobility while temporarily lowering penalties.
    # The dip is centered after the constant phase, when wakes have moved enough
    # for a controlled escape but the run still has time to re-feasibilize.
    pulse = jnp.exp(-((t - 0.48) / 0.095) ** 2)
    late_pulse = jnp.exp(-((t - 0.66) / 0.075) ** 2)

    lr = base_lr * (1.0 + 0.45 * pulse + 0.16 * late_pulse)

    alpha = base_alpha * (1.0 - 0.52 * jnp.exp(-((t - 0.48) / 0.095) ** 2))
    alpha = alpha * (1.0 - 0.18 * late_pulse)

    # Restore extra constraint authority late, when LR is already small.
    polish = 1.0 / (1.0 + jnp.exp(-28.0 * (t - 0.78)))
    alpha = alpha * (1.0 + 0.28 * polish)
    alpha = jnp.maximum(alpha, 0.20 * alpha0)

    beta1 = 12.0 / 100.0
    beta2 = 25.0 / 100.0

    return lr, alpha, beta1, beta2
