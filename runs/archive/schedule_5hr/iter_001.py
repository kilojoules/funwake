"""Cosine annealing schedule with warmup and three-phase momentum.

Improvements over seed:
- Cosine decay (smoother than 1/(1+mid*t))
- Short warmup phase for stable early gradients
- Higher beta1 in exploration phase for momentum-driven exploration
- Aggressive alpha ramp in final polishing phase
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    """Three-phase cosine annealing schedule.

    Phase 1 (0-5%): Warmup - ramp LR from 0.1*lr0 to lr0
    Phase 2 (5-70%): Main optimization - cosine decay to 0.05*lr0
    Phase 3 (70-100%): Polishing - low LR, high alpha

    Args:
        step: current iteration (0 to total_steps-1)
        total_steps: total number of iterations
        lr0: initial learning rate
        alpha0: initial penalty weight

    Returns:
        (lr, alpha, beta1, beta2)
    """
    t = step / jnp.maximum(total_steps, 1.0)

    # Phase boundaries
    warmup_end = 0.05
    main_end = 0.70

    # === Learning rate schedule ===
    # Warmup: linear ramp from 0.1*lr0 to lr0
    lr_warmup = lr0 * (0.1 + 0.9 * t / warmup_end)

    # Main: cosine decay from lr0 to 0.05*lr0
    main_progress = (t - warmup_end) / (main_end - warmup_end)
    lr_main = lr0 * (0.05 + 0.95 * 0.5 * (1 + jnp.cos(jnp.pi * main_progress)))

    # Polish: decay to very low LR for constraint enforcement
    polish_progress = (t - main_end) / (1.0 - main_end)
    lr_polish = lr0 * (0.001 + 0.049 * (1 - polish_progress))

    lr = jnp.where(t < warmup_end, lr_warmup,
         jnp.where(t < main_end, lr_main, lr_polish))

    # === Alpha (penalty weight) schedule ===
    # Ramps up as LR decays, with extra boost in polishing phase
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    # Extra 3x multiplier during polishing for strict constraint enforcement
    alpha_boost = jnp.where(t >= main_end, 3.0, 1.0)
    alpha = base_alpha * alpha_boost

    # === Momentum schedule ===
    # Higher momentum early for exploration, lower for polishing
    beta1 = jnp.where(t < warmup_end, 0.3,
            jnp.where(t < main_end, 0.2, 0.1))

    beta2 = jnp.where(t < warmup_end, 0.4,
            jnp.where(t < main_end, 0.3, 0.2))

    return lr, alpha, beta1, beta2
