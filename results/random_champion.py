"""Random search sample 30.

Family params: {"lr_mult": 4.409, "lr_final_ratio": "0.002411", "warmup_frac": 0.05, "decay": "polynomial", "perturbation": "gaussian_bumps", "alpha": "coupled", "alpha_final_mult": 1741.4, "beta1": 0.196, "beta2": 0.7677}
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Learning rate base curve
    lr_init = 4.4091 * lr0
    lr_final_ratio = 0.00241104
    lr_min = lr_init * lr_final_ratio

    warmup_frac = 0.05
    warmup_lr = lr_init * jnp.minimum(1.0, t / jnp.maximum(warmup_frac, 1e-6))

    post_warmup_t = jnp.maximum(t - warmup_frac, 0.0) / jnp.maximum(1.0 - warmup_frac, 1e-6)

    lr_decay = lr_min + (lr_init - lr_min) * (1.0 - post_warmup_t) ** 3.75

    lr_base = jnp.where(t < warmup_frac, warmup_lr, lr_decay)

    # Perturbation
    perturbation = 0.422 * jnp.exp(-0.5 * ((t - 0.573) / 0.061)**2)

    lr = lr_base * (1.0 + perturbation)
    lr = jnp.maximum(lr, 1e-10)

    # Alpha: penalty weight
    alpha = 9.15 * alpha0 * lr0 / lr

    beta1 = 0.196
    beta2 = 0.7677

    return lr, alpha, beta1, beta2
