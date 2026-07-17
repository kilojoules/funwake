"""Single Gaussian bump with very late alpha-only repair."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.9814 * lr0
    lr_final_ratio = 0.027468
    lr_min = lr_init * lr_final_ratio

    warmup_frac = 0.02
    warmup_lr = lr_init * jnp.minimum(1.0, t / jnp.maximum(warmup_frac, 1e-6))
    post_warmup_t = jnp.maximum(t - warmup_frac, 0.0) / jnp.maximum(
        1.0 - warmup_frac, 1e-6
    )
    lr_decay = lr_init * jnp.exp(-3.03 * post_warmup_t) + lr_min
    lr_base = jnp.where(t < warmup_frac, warmup_lr, lr_decay)

    bump = 0.182 * jnp.exp(-0.5 * ((t - 0.653) / 0.059) ** 2)
    lr = jnp.maximum(lr_base * (1.0 + bump), 1e-10)

    repair = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.92) / 0.08))
    repair = repair * repair * (3.0 - 2.0 * repair)

    alpha = 7.59 * alpha0 * lr0 / lr
    alpha = alpha * (1.0 + 45.0 * repair + 1200.0 * repair * repair)

    beta1 = 0.16
    beta2 = 0.5427

    return lr, alpha, beta1, beta2
