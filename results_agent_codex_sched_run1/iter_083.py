"""HYPOTHESIS: The rounded transferred warmup dual-bump schedule should keep
its basin, but lowering beta2 only inside the two LR mobility windows may make
Adam react more directly to fresh wake/constraint gradients without the
boundary drift caused by extra alpha pulses.
AXIS: rounded_dual_bump_beta2_window_gating
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    K = 4.327099
    logM = 3.420468
    W = 0.061433
    A1 = 0.165784
    A2 = 0.458205
    c1 = 0.923544
    c2 = 0.546162
    w1 = 0.112811
    w2 = 0.120347
    C = 2.879478
    D = 16.850946
    B1 = 0.239994
    B2 = 0.635963

    lr_init = K * lr0
    lr_min = lr_init / (10.0 ** logM)

    warmup_lr = lr_init * t / jnp.maximum(W, 1e-6)
    cosine_t = (t - W) / jnp.maximum(1.0 - W, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < W, warmup_lr, cosine_lr)

    bump1_shape = jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
    bump2_shape = jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
    lr = jnp.maximum(lr_base + A1 * lr_init * bump1_shape + A2 * lr_init * bump2_shape, 1e-10)

    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = C * alpha0 * lr_init / lr + D * alpha0 * late**2

    gate = jnp.maximum(bump1_shape, bump2_shape)
    beta2 = B2 - 0.10 * gate

    return lr, alpha, B1, beta2
