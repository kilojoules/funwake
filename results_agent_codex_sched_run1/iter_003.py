"""HYPOTHESIS: A very shallow anti-phase dip plus strong late penalty dominance should keep the mobility benefit without leaving residual boundary violations.
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

    # Very low final objective LR lets the amplified late penalty clean up
    # small boundary residuals without much renewed wake-seeking motion.
    mid = 499.0 / decay_steps
    base_lr = jnp.where(decaying, lr0 / (1.0 + mid * decay_step), lr0)
    base_alpha = jnp.where(
        decaying,
        alpha0 * lr0 / jnp.maximum(base_lr, 1e-10),
        alpha0,
    )

    pulse = jnp.exp(-((t - 0.44) / 0.070) ** 2)
    lr = base_lr * (1.0 + 0.12 * pulse)

    alpha = base_alpha * (1.0 - 0.12 * jnp.exp(-((t - 0.44) / 0.070) ** 2))

    recovery = 1.0 / (1.0 + jnp.exp(-22.0 * (t - 0.55)))
    final_polish = 1.0 / (1.0 + jnp.exp(-32.0 * (t - 0.76)))
    alpha = alpha * (1.0 + 2.0 * recovery + 2.0 * final_polish)
    alpha = jnp.maximum(alpha, 0.50 * alpha0)

    beta1 = 11.0 / 100.0
    beta2 = 25.0 / 100.0

    return lr, alpha, beta1, beta2
