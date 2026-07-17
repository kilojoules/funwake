"""Constant-LR search with nonlinear penalty tightening.

HYPOTHESIS: A fixed moderate step size can keep the wake layout mobile for the
full run, while a delayed high-curvature alpha ramp shifts the same motion from
objective exploration to constraint repair near the end.
AXIS: lr_constant with late nonlinear alpha repair and TopFarm low-momentum
Adam.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr = 0.16 * lr0

    early = jnp.minimum(1.0, t / 0.18)
    early = early * early * (3.0 - 2.0 * early)
    mid = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.28) / 0.42))
    mid = mid * mid * (3.0 - 2.0 * mid)
    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.70) / 0.30))
    late = late * late * (3.0 - 2.0 * late)

    alpha = alpha0 * (3.5 + 14.0 * early + 38.0 * mid + 820.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
