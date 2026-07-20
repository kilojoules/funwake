"""HYPOTHESIS: Keeping the best LR/alpha path but making Adam more reactive only in the final penalty-dominated polish may reduce residual constraint/objective lag without disrupting the productive basin.
AXIS: late_adam_reactivity_gate
LESSON: Pending score.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    lr_base = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    lr = lr_base + 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.692) / 0.05) ** 2)

    alpha = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    alpha = alpha + 3.0 * alpha0 * (jnp.maximum(t - 0.5, 0.0) / 0.5) ** 2

    gate = 1.0 / (1.0 + jnp.exp(-34.0 * (t - 0.82)))
    beta1 = 0.3 - 0.10 * gate
    beta2 = 0.5 - 0.12 * gate

    return lr, alpha, beta1, beta2
