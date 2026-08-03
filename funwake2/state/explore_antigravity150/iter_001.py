import jax.numpy as jnp

# diameter-rule constant: c = (DEI best lr0) / (DEI D) = 200 / 240
_C = 200.0 / 240.0

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Safely cast to JAX arrays to ensure traceability without Python branching or casting
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    D_f = jnp.asarray(D, dtype=jnp.float32)
    g_min = jnp.asarray(gamma_min, dtype=jnp.float32)

    # Fractional progress [0.0, 1.0]
    frac = step_f / jnp.maximum(total_f, 1.0)

    # 1. Learning Rate: WSD (Warmup, Stable, Decay) schedule
    # 0.0 -> 0.5: Stable exploration phase held at lr0 (push AEP higher)
    # 0.5 -> 0.9: Cosine decay down to gamma_min
    # 0.9 -> 1.0: Terminal cooldown held at gamma_min
    lr0 = _C * D_f
    decay_progress = jnp.clip((frac - 0.5) / 0.4, 0.0, 1.0)
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
    lr_decayed = g_min + (lr0 - g_min) * cosine_mult
    
    lr = jnp.where(
        frac < 0.5, 
        lr0,
        jnp.where(frac < 0.9, lr_decayed, g_min)
    )

    # 2. Alpha (Penalty): Delayed ramp + Terminal spike
    # 0.0 -> 0.5: Delayed ramp, hold at alpha0 for unhindered exploration
    # 0.5 -> 0.9: Ramp to a bounded plateau (10 * alpha0) to gently enforce constraints
    # 0.9 -> 1.0: Terminal feasibility spike (100 * alpha0) to guarantee strictly feasible layouts
    alpha_ramp = alpha0 * (1.0 + 9.0 * decay_progress)
    alpha = jnp.where(
        frac < 0.5, 
        alpha0,
        jnp.where(frac < 0.9, alpha_ramp, 100.0 * alpha0)
    )

    # 3. Betas: Phase-transition with the alpha phase
    # beta1: Drop momentum during the terminal spike to avoid constraint overshoot
    beta1 = jnp.where(frac < 0.9, 0.1, 0.05)
    
    # beta2: Ramp up to rectify adaptive-lr variance and absorb constraint-curvature
    beta2 = jnp.where(
        frac < 0.5, 
        0.2,
        jnp.where(frac < 0.9, 0.5, 0.99)
    )

    return lr, alpha, beta1, beta2