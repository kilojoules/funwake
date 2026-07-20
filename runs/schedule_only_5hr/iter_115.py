"""Iter 115: 3 stages: explore/refine/lock. Cosine LR, 4x, lr/10000."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    # 3 stages for betas
    # Explore (0-30%): moderate momentum (0.3, 0.5)
    # Refine (30-70%): lower momentum (0.15, 0.3)
    # Lock (70-100%): lowest momentum (0.05, 0.1)
    stage1 = t < 0.3
    stage2 = (t >= 0.3) & (t < 0.7)
    beta1 = jnp.where(stage1, 0.3, jnp.where(stage2, 0.15, 0.05))
    beta2 = jnp.where(stage1, 0.5, jnp.where(stage2, 0.3, 0.1))

    return lr, alpha, beta1, beta2
