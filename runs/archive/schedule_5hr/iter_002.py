"""Four-phase schedule: warmup, explore, refine, enforce.

Key changes from iter_001:
- Higher momentum (beta1=0.5, beta2=0.6) during exploration for faster convergence
- Longer exploration phase (50%) for better AEP
- Very aggressive constraint enforcement in final 20% (10x alpha boost)
- No warmup waste - immediate ramp to full LR
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps, 1.0)

    # Phase boundaries
    warmup_end = 0.02
    explore_end = 0.50
    refine_end = 0.80

    # === Learning rate ===
    # Warmup: quick ramp
    lr_warmup = lr0 * (0.2 + 0.8 * t / warmup_end)

    # Explore: constant at lr0
    lr_explore = lr0

    # Refine: cosine decay from lr0 to 0.05*lr0
    refine_progress = (t - explore_end) / (refine_end - explore_end)
    lr_refine = lr0 * (0.05 + 0.95 * 0.5 * (1 + jnp.cos(jnp.pi * refine_progress)))

    # Enforce: linear decay from 0.05*lr0 to 0.001*lr0
    enforce_progress = (t - refine_end) / (1.0 - refine_end)
    lr_enforce = lr0 * (0.001 + 0.049 * (1 - enforce_progress))

    lr = jnp.where(t < warmup_end, lr_warmup,
         jnp.where(t < explore_end, lr_explore,
         jnp.where(t < refine_end, lr_refine, lr_enforce)))

    # === Alpha (penalty weight) ===
    base_alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    # 10x boost in enforcement phase
    boost = jnp.where(t >= refine_end, 10.0,
            jnp.where(t >= explore_end, 2.0, 1.0))
    alpha = base_alpha * boost

    # === Momentum ===
    # Higher momentum during exploration for AEP, lower for constraint enforcement
    beta1 = jnp.where(t < explore_end, 0.5,
            jnp.where(t < refine_end, 0.3, 0.1))

    beta2 = jnp.where(t < explore_end, 0.6,
            jnp.where(t < refine_end, 0.4, 0.2))

    return lr, alpha, beta1, beta2
