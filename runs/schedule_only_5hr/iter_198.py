"""Iter 198: Dynamic beta scheduling — high momentum early, low late.

Beta1 starts at 0.9 (strong momentum for exploration) and decays to 0.1
(responsive to local gradients for fine-tuning).
Beta2 starts at 0.999 (standard Adam) and decays to 0.2 (TopFarm-style).
This is fundamentally different from all prior attempts which used fixed betas.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine (proven base from iter_179)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    # Bump at 0.7
    bump = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.7) / 0.05) ** 2)
    lr = lr_base + bump

    # Alpha: same proven structure
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    # Dynamic betas: high momentum early -> low momentum late
    beta1 = 0.9 - 0.8 * t  # 0.9 -> 0.1
    beta2 = 0.999 - 0.799 * t  # 0.999 -> 0.2

    return lr, alpha, beta1, beta2
