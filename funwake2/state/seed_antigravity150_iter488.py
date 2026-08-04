import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceability; cast to JAX arrays to prevent float() tracing errors
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total_steps_f = jnp.asarray(total_steps, dtype=jnp.float32)
    progress = jnp.clip(step_f / jnp.maximum(total_steps_f, 1.0), 0.0, 1.0)
    
    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    a0 = jnp.asarray(alpha0, dtype=jnp.float32)

    # WSD (Warmup, Stable, Decay) Schedule + Terminal Feasibility Phase
    # Phase boundaries: 10%, 50%, 90%, 100%
    
    # 1. Warmup (0% - 10%)
    p_w = jnp.clip(progress / 0.10, 0.0, 1.0)
    lr_w = 0.05 * D_f + (1.50 * D_f - 0.05 * D_f) * p_w
    alpha_w = a0 * 0.1
    beta1_w = 0.20
    beta2_w = 0.20 + 0.20 * p_w  # 0.20 -> 0.40
    
    # 2. Stable / Hold (10% - 50%)
    p_s = jnp.clip((progress - 0.10) / 0.40, 0.0, 1.0)
    lr_s = 1.50 * D_f - (1.50 * D_f - 1.00 * D_f) * p_s  # Gentle decline for exploration
    alpha_s = a0 * 0.1  # Maintain loose penalty for unconstrained exploration
    beta1_s = 0.20
    beta2_s = 0.40
    
    # 3. Decay & Delayed Logistic Penalty Ramp (50% - 90%)
    p_d = jnp.clip((progress - 0.50) / 0.40, 0.0, 1.0)
    # Cosine decay for learning rate
    lr_d = 0.05 * D_f + (1.00 * D_f - 0.05 * D_f) * 0.5 * (1.0 + jnp.cos(jnp.pi * p_d))
    
    # Normalized logistic ramp for alpha, centered at p_d = 0.5 (70% total progress)
    raw_logistic = 1.0 / (1.0 + jnp.exp(-15.0 * (p_d - 0.5)))
    min_log = 1.0 / (1.0 + jnp.exp(7.5))
    max_log = 1.0 / (1.0 + jnp.exp(-7.5))
    logistic = (raw_logistic - min_log) / (max_log - min_log)
    alpha_d = a0 * 0.1 + (a0 * 30.0 - a0 * 0.1) * logistic
    
    # Beta1 drops to reduce momentum on penalty gradients; Beta2 ramps up to absorb curvature
    beta1_d = 0.20 - 0.15 * p_d  # 0.20 -> 0.05
    beta2_d = 0.40 + 0.50 * p_d  # 0.40 -> 0.90
    
    # 4. Terminal Feasibility Spike (90% - 100%)
    p_t = jnp.clip((progress - 0.90) / 0.10, 0.0, 1.0)
    # Drop LR exactly to tolerance
    lr_t = gamma_min_f + (0.05 * D_f - gamma_min_f) * ((1.0 - p_t) ** 2.0)
    
    # Spike penalty to mathematically guarantee feasibility
    alpha_term_val = a0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha_t = a0 * 30.0 + (alpha_term_val - a0 * 30.0) * (p_t ** 2.0)
    
    beta1_t = 0.05 - 0.03 * p_t  # 0.05 -> 0.02
    beta2_t = 0.90 + 0.08 * p_t  # 0.90 -> 0.98

    # Combine phases
    lr = jnp.where(progress < 0.10, lr_w,
         jnp.where(progress < 0.50, lr_s,
         jnp.where(progress < 0.90, lr_d, lr_t)))

    alpha = jnp.where(progress < 0.10, alpha_w,
            jnp.where(progress < 0.50, alpha_s,
            jnp.where(progress < 0.90, alpha_d, alpha_t)))

    beta1 = jnp.where(progress < 0.10, beta1_w,
            jnp.where(progress < 0.50, beta1_s,
            jnp.where(progress < 0.90, beta1_d, beta1_t)))

    beta2 = jnp.where(progress < 0.10, beta2_w,
            jnp.where(progress < 0.50, beta2_s,
            jnp.where(progress < 0.90, beta2_d, beta2_t)))

    return lr, alpha, beta1, beta2