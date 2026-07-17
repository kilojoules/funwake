"""Late objective-polish shelf on the proven noisy repair backbone.

HYPOTHESIS: The incumbent repeatedly reaches the same feasible basin, but its
late inverse-LR penalty may become dominant before the objective has finished
separating wake interactions. A narrow post-noise LR shelf with a tiny alpha
softening should allow one more objective polish, followed by the same strong
repair finish for feasibility.
AXIS: late LR shelf plus brief penalty softening on incumbent noisy envelope.
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

    # After the stochastic window fades, keep a small deterministic movement
    # shelf so objective gradients can still rearrange close turbine pairs.
    polish = jnp.exp(-0.5 * ((t - 0.785) / 0.055) ** 2)
    polish_gate = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.705) / 0.055))
    polish_gate = polish_gate * polish_gate * (3.0 - 2.0 * polish_gate)
    polish_end = jnp.minimum(1.0, jnp.maximum(0.0, (0.885 - t) / 0.055))
    polish_end = polish_end * polish_end * (3.0 - 2.0 * polish_end)
    polish = polish * polish_gate * polish_end
    lr = jnp.maximum(lr, lr0 * (0.00055 + 0.020 * polish))
    lr = jnp.maximum(lr, lr_floor)

    alpha = 20.0 * alpha0 * lr0 / jnp.maximum(lr_base, 1e-10)
    alpha = alpha * (1.0 + 0.35 * t)
    alpha = alpha * (1.0 - 0.030 * polish)

    late = jnp.minimum(1.0, jnp.maximum(0.0, (t - 0.68) / 0.32))
    late = late * late * (3.0 - 2.0 * late)
    alpha = alpha * (1.0 + 7.0 * late + 52.0 * late * late)

    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2
