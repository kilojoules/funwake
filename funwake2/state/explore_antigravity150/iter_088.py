import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Multi-Cycle Learning Rate ---
    # Three cosine-annealing cycles with warm restarts.
    # Each restart gives the optimizer momentum to escape local boundary traps.
    c1_end = 0.45
    c2_end = 0.75
    c3_end = 0.93
    
    # Progress within each cycle [0, 1]
    p_c1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p_c2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p_c3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    # Cosine multiplier: goes from 1.0 down to 0.0
    cos_1 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c1))
    cos_2 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c2))
    cos_3 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c3))
    
    # Decreasing peaks for learning rate to transition from global to local search
    lr_max_1 = 1.35 * D_f
    lr_max_2 = 0.50 * D_f
    lr_max_3 = 0.15 * D_f
    
    lr_1 = gamma_min_f + (lr_max_1 - gamma_min_f) * cos_1
    lr_2 = gamma_min_f + (lr_max_2 - gamma_min_f) * cos_2
    lr_3 = gamma_min_f + (lr_max_3 - gamma_min_f) * cos_3
    
    lr_main = jnp.where(progress < c1_end, lr_1,
                jnp.where(progress < c2_end, lr_2, lr_3))
                
    is_terminal = progress >= c3_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha with Mid-Run Restarts ---
    # Alpha rises during each cycle to enforce feasibility (restoration bursts), 
    # then drops at the restart to allow the layout to break out and explore again.
    
    # Cycle 1: very soft start, medium peak
    alpha_base_1 = alpha0 * 0.05
    alpha_peak_1 = alpha0 * 8.0
    
    # Cycle 2: moderate start, high peak
    alpha_base_2 = alpha0 * 1.0
    alpha_peak_2 = alpha0 * 15.0
    
    # Cycle 3: strict start, very high peak
    alpha_base_3 = alpha0 * 5.0
    alpha_peak_3 = alpha0 * 30.0
    
    # Alpha grows opposite to LR: as LR cosine-decays, Alpha cosine-grows
    alpha_1 = alpha_peak_1 - (alpha_peak_1 - alpha_base_1) * cos_1
    alpha_2 = alpha_peak_2 - (alpha_peak_2 - alpha_base_2) * cos_2
    alpha_3 = alpha_peak_3 - (alpha_peak_3 - alpha_base_3) * cos_3
    
    alpha_main = jnp.where(progress < c1_end, alpha_1,
                   jnp.where(progress < c2_end, alpha_2, alpha_3))
                   
    # Terminal feasibility spike ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # High momentum (low beta2) at cycle starts for rapid layout shifting.
    # Damping (high beta2) as cycle ends to absorb constraint curvature.
    
    b1_start, b1_end = 0.15, 0.04
    b2_start, b2_end = 0.10, 0.85
    
    b1_1 = b1_end + (b1_start - b1_end) * cos_1
    b1_2 = b1_end + (b1_start - b1_end) * cos_2
    b1_3 = b1_end + (b1_start - b1_end) * cos_3
    
    b2_1 = b2_end + (b2_start - b2_end) * cos_1
    b2_2 = b2_end + (b2_start - b2_end) * cos_2
    b2_3 = b2_end + (b2_start - b2_end) * cos_3
    
    beta1_main = jnp.where(progress < c1_end, b1_1,
                   jnp.where(progress < c2_end, b1_2, b1_3))
    beta2_main = jnp.where(progress < c1_end, b2_1,
                   jnp.where(progress < c2_end, b2_2, b2_3))
                   
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2