import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Two-Cycle SGDR (Warm Restarts) with Feasibility Bursts ---
    # We replace the single WSD cooldown with a multi-cycle approach.
    # Each cycle starts with high LR / low penalty for exploration,
    # and ends with low LR / high penalty for a feasibility-restoration burst.
    # The warm restart acts as a shock that knocks the layout out of 
    # constrained local minima, funnelling toward strict feasibility.
    
    phase1_end = 0.60
    phase2_end = 0.90
    
    # Cosine scheduling helpers
    b1_start, b1_end = 0.12, 0.04
    b2_start, b2_end = 0.15, 0.85
    
    # -- Phase 1: Deep Exploration (0.0 to 0.6) --
    t_phase1 = jnp.clip(progress / phase1_end, 0.0, 1.0)
    
    lr_max1 = 1.80 * D_f  # Higher initial peak for early layout shifting
    alpha_min1 = alpha0_f * 0.1
    alpha_max1 = alpha0_f * 15.0
    
    cos_t1 = jnp.cos(jnp.pi * t_phase1)
    lr1 = gamma_min_f + 0.5 * (lr_max1 - gamma_min_f) * (1.0 + cos_t1)
    alpha1 = alpha_min1 + 0.5 * (alpha_max1 - alpha_min1) * (1.0 - cos_t1)
    
    b1_1 = b1_end + 0.5 * (b1_start - b1_end) * (1.0 + cos_t1)
    b2_1 = b2_end + 0.5 * (b2_start - b2_end) * (1.0 + cos_t1)
    
    # -- Phase 2: Warm Restart & Fine-Tuning (0.6 to 0.9) --
    t_phase2 = jnp.clip((progress - phase1_end) / (phase2_end - phase1_end), 0.0, 1.0)
    
    lr_max2 = 0.90 * D_f  # Reduced exploration envelope
    alpha_min2 = alpha0_f * 1.0  # Stricter baseline penalty 
    alpha_max2 = alpha0_f * 30.0 # Higher plateau for final convergence
    
    cos_t2 = jnp.cos(jnp.pi * t_phase2)
    lr2 = gamma_min_f + 0.5 * (lr_max2 - gamma_min_f) * (1.0 + cos_t2)
    alpha2 = alpha_min2 + 0.5 * (alpha_max2 - alpha_min2) * (1.0 - cos_t2)
    
    b1_2 = b1_end + 0.5 * (b1_start - b1_end) * (1.0 + cos_t2)
    b2_2 = b2_end + 0.5 * (b2_start - b2_end) * (1.0 + cos_t2)
    
    # -- Combine Phases --
    is_phase2 = progress > phase1_end
    lr_main = jnp.where(is_phase2, lr2, lr1)
    alpha_main = jnp.where(is_phase2, alpha2, alpha1)
    beta1_main = jnp.where(is_phase2, b1_2, b1_1)
    beta2_main = jnp.where(is_phase2, b2_2, b2_1)
    
    # -- Terminal Phase: Strict Feasibility (0.9 to 1.0) --
    # Massive spike in penalty and drop in LR/momentum to guarantee compliance.
    is_terminal = progress >= phase2_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2