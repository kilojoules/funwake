"""TopFarm-style low momentum with extended constant-LR and aggressive enforcement.

Key insight: topfarm_sgd_solve uses beta1=0.1, beta2=0.2 for good reason -
the AEP landscape is noisy. Higher momentum overshoots.

Strategy: Long constant LR phase (60%), fast decay (20%), brutal enforcement (20%).
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    # Phase boundaries
    const_end = 0.60    # 60% at constant LR (exploration)
    decay_end = 0.80    # 20% cosine decay (refinement)
    # Last 20%: enforcement

    # === Learning rate ===
    # Constant phase: full lr0
    lr_const = lr0

    # Decay phase: cosine from lr0 to 0.02*lr0
    decay_progress = (t - const_end) / (decay_end - const_end)
    lr_decay = lr0 * (0.02 + 0.98 * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress)))

    # Enforcement: exponential decay to very low LR
    enforce_progress = (t - decay_end) / (1.0 - decay_end)
    lr_enforce = lr0 * 0.02 * jnp.exp(-4.0 * enforce_progress)

    lr = jnp.where(t < const_end, lr_const,
         jnp.where(t < decay_end, lr_decay, lr_enforce))

    # === Alpha ===
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    # Massive boost in enforcement phase
    boost = jnp.where(t >= decay_end, 50.0,
            jnp.where(t >= const_end, 1.0, 1.0))
    alpha = base_alpha * boost

    # === Momentum: TopFarm-style low values ===
    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
