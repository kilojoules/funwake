"""Tight guarded LR shelf with stronger final repair.

HYPOTHESIS: The guarded shelf improved objective value but remained just
infeasible, so reduce shelf amplitude and pair it with a materially stronger
local alpha guard plus a slightly firmer final repair. This tests whether the
late mobility gain can be retained without giving up feasibility.
AXIS: constrained late LR shelf with strengthened penalty repair.
LESSON: pending
"""
import jax
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step.astype(float) / jnp.maximum(total_steps - 1.0, 1.0)

    lr_start = 4.0 * lr0
    lr_floor = 0.00038 * lr0
    lr_base = jnp.maximum(lr_start * jnp.exp(-8.0 * t * t), lr_floor)

    key = jax.random.fold_in(jax.random.PRNGKey(22022), step.astype(jnp.uint32))
    lr_noise = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    rise = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.12) / 0.18))
    fall = jnp.minimum(1.0, jnp.maximum(0.0, (0.72 - t) / 0.22))
    noise_gate = rise * rise * (3.0 - 2.0 * rise)
    noise_gate = noise_gate * fall * fall * (3.0 - 2.0 * fall)

    lr = lr_base + lr_base * (0.08 * noise_gate) * lr_noise

    polish = jnp.exp(-0.5 * ((t - 0.765) / 0.045) ** 2)
    polish_gate = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.705) / 0.045))
    polish_gate = polish_gate * polish_gate * (3.0 - 2.0 * polish_gate)
    polish_end = jnp.minimum(1.0, jnp.maximum(0.0, (0.845 - t) / 0.045))
    polish_end = polish_end * polish_end * (3.0 - 2.0 * polish_end)
    polish = polish * polish_gate * polish_end
    lr = jnp.maximum(lr, lr0 * (0.00050 + 0.0090 * polish))
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)
    alpha = alpha * (1.0 + 0.22 * polish)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.66) / 0.34))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.8 * late + 58.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
