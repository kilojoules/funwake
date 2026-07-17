"""HYPOTHESIS: The best train path has saturated under static Adam moments.
Keep its LR/penalty envelope, but use short momentum reset windows during the
two mobility bumps so the optimizer can respond to changed wake gradients,
then increase damping only in the terminal feasibility closure.
AXIS: dual_window_momentum_reset_terminal_damping
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.327099 * lr0
    lr_min = lr_init / (10.0 ** 3.420468)

    warmup_end = 0.061433
    warmup_lr = lr_init * t / jnp.maximum(warmup_end, 1e-6)
    cosine_t = (t - warmup_end) / jnp.maximum(1.0 - warmup_end, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    mid_shape = jnp.exp(-0.5 * ((t - 0.546162) / 0.120347) ** 2)
    late_shape = jnp.exp(-0.5 * ((t - 0.923544) / 0.112811) ** 2)
    lr = lr_base + 0.458205 * lr_init * mid_shape + 0.165784 * lr_init * late_shape
    lr = jnp.maximum(lr, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = 2.879478 * alpha0 * lr_init / lr
    alpha = alpha + 16.850946 * alpha0 * late**2

    terminal = jnp.clip((t - 0.94) / 0.06, 0.0, 1.0)
    terminal = terminal * terminal * (3.0 - 2.0 * terminal)
    reset = jnp.maximum(0.70 * mid_shape, 0.95 * late_shape)

    beta1 = 0.265 - 0.075 * reset + 0.035 * terminal
    beta2 = 0.665 - 0.115 * reset + 0.045 * terminal
    beta1 = jnp.clip(beta1, 0.16, 0.32)
    beta2 = jnp.clip(beta2, 0.50, 0.74)

    return lr, alpha, beta1, beta2
