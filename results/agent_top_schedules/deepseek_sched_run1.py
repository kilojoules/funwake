import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    """66.4% warmup, cosine decay to lr0/500, cubic alpha to 21000x, beta2=0.97."""
    warmup_frac = 0.664
    warmup_steps = int(warmup_frac * total_steps)
    decay_steps = total_steps - warmup_steps

    decaying = step >= warmup_steps
    decay_frac = (step - warmup_steps) / jnp.maximum(decay_steps - 1, 1)
    decay_frac = jnp.clip(decay_frac, 0.0, 1.0)
    
    lr_min = lr0 / 500.0
    lr = jnp.where(decaying,
                   lr_min + 0.5 * (lr0 - lr_min) * (1.0 + jnp.cos(jnp.pi * decay_frac)),
                   lr0)

    total_frac = step / total_steps
    alpha = alpha0 * (1.0 + 20999.0 * total_frac**3)

    beta1 = 0.9
    beta2 = 0.97

    return lr, alpha, beta1, beta2