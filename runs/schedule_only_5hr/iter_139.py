"""Iter 139: Step-function LR with smooth transitions.

3 plateaus connected by smooth sigmoid transitions. Spends more time
at each plateau (exploring vs refining vs converging) than cosine.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Sigmoid transition function
    def sigmoid(x, center, sharpness):
        return 1.0 / (1.0 + jnp.exp(-sharpness * (x - center)))

    # Three plateaus: lr_init (0-30%), 0.15*lr_init (30-70%), lr_min (70-100%)
    lr_mid = 0.15 * lr_init
    drop1 = sigmoid(t, 0.30, 40.0)  # sharp drop at t=0.30
    drop2 = sigmoid(t, 0.70, 40.0)  # sharp drop at t=0.70

    lr = lr_init * (1.0 - drop1) + lr_mid * (drop1 - drop2) + lr_min * drop2

    alpha = alpha0 * lr_init / jnp.maximum(lr, 1e-10)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
