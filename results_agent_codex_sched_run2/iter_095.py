"""Long constant search with dual LR kicks and late penalty snap.

HYPOTHESIS: The recent noisy-exponential schedules keep returning the same
5564.5 basin. A different regime can spend most of the run at a stable LR,
use two deterministic basin-change kicks while Adam's second moment is slow,
then switch to a low-memory repair tail with a strong but delayed penalty snap.
AXIS: constant_phase_dual_kick_with_beta2_switch_and_late_repair
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    warmup = jnp.clip(t / 0.05, 0.0, 1.0)
    decay_start = 0.625
    decay_prog = jnp.clip((t - decay_start) / (1.0 - decay_start), 0.0, 1.0)

    lr_base = jnp.where(
        t < decay_start,
        lr0,
        lr0 * (1.0 - decay_prog),
    )
    lr_base = jnp.maximum(lr_base, 0.001 * lr0)

    kick1 = 5.0 * jnp.exp(-0.5 * ((t - 0.25) / 0.05) ** 2)
    kick2 = 4.0 * jnp.exp(-0.5 * ((t - 0.50) / 0.05) ** 2)
    lr = warmup * lr_base * (1.0 + kick1 + kick2)
    lr = jnp.maximum(lr, 1e-10)

    alpha = alpha0 * lr0 / lr
    alpha = alpha * (1.0 + 9.0 * t * t)

    repair = jnp.clip((t - 0.80) / 0.20, 0.0, 1.0)
    alpha = alpha * (1.0 + 500.0 * repair + 3000.0 * repair**3)

    beta1 = 0.1
    beta2 = jnp.where(t < decay_start, 0.9, 0.2)

    return lr, alpha, beta1, beta2
