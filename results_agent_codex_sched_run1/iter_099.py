"""HYPOTHESIS: The rounded dual-bump path finds the best basin, but the
terminal penalty-dominated tail may slightly over-polish feasibility after the
useful wake movement is complete. Freezing the final short tail tests whether
preserving the post-bump layout improves AEP without changing the discovery
phase.
AXIS: dual_bump_terminal_freeze
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

    bump1 = A1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
    bump2 = A2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
    lr_active = jnp.maximum(lr_base + bump1 + bump2, 1e-10)
    lr = jnp.where(t < 0.982, lr_active, 0.0)

    alpha_base = C * alpha0 * lr_init / lr_active
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha = alpha_base + D * alpha0 * late**2

    return lr, alpha, B1, B2
