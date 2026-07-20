"""HYPOTHESIS: A smaller anti-phase alpha dip can preserve the escape mechanism while a lower final objective LR and earlier alpha recovery repair boundary drift.
AXIS: alpha_anti_phase_dip
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    denom = jnp.maximum(total_steps - 1, 1)
    t = step / denom

    const_phase = total_steps // 3
    decay_steps = jnp.maximum(total_steps - const_phase, 1.0)
    decay_step = jnp.maximum(step - const_phase, 0.0)
    decaying = step >= const_phase

    # Decay to 0.5% lr0 so late objective motion is weaker than constraint repair.
    mid = 199.0 / decay_steps
    base_lr = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)
    base_alpha = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(base_lr, 1e-10),
        alpha0,
    )

    pulse = jnp.exp(-((t - 0.46) / 0.080) ** 2)
    settle_pulse = jnp.exp(-((t - 0.62) / 0.060) ** 2)

    lr = base_lr * (1.0 + 0.24 * pulse + 0.08 * settle_pulse)

    alpha = base_alpha * (1.0 - 0.28 * jnp.exp(-((t - 0.46) / 0.080) ** 2))
    alpha = alpha * (1.0 - 0.07 * settle_pulse)

    recovery = 1.0 / (1.0 + jnp.exp(-24.0 * (t - 0.66)))
    final_polish = 1.0 / (1.0 + jnp.exp(-34.0 * (t - 0.84)))
    alpha = alpha * (1.0 + 0.95 * recovery + 0.45 * final_polish)
    alpha = jnp.maximum(alpha, 0.35 * alpha0)

    beta1 = 12.0 / 100.0
    beta2 = 25.0 / 100.0

    return lr, alpha, beta1, beta2
