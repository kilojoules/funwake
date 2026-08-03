import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f
    
    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # --- 1. SGDR with Warm Restarts ---
    # Multi-cycle cosine annealing. Each cycle starts with a high LR
    # to forcefully pop out of local minima, then smoothly cools down.
    p_c1, p_c2, p_c3 = 0.35, 0.70, 0.90
    
    def cosine_cycle(p, p_start, p_end, lr_peak, lr_min):
        p_norm = jnp.clip((p - p_start) / (p_end - p_start), 0.0, 1.0)
        return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * p_norm))
        
    lr_c1 = cosine_cycle(progress, 0.00, p_c1, 1.40 * D_f, gamma_min_f * 10.0)
    lr_c2 = cosine_cycle(progress, p_c1, p_c2, 0.90 * D_f, gamma_min_f * 5.0)
    lr_c3 = cosine_cycle(progress, p_c2, p_c3, 0.40 * D_f, gamma_min_f)
    
    lr_main = jnp.where(progress < p_c1, lr_c1,
              jnp.where(progress < p_c2, lr_c2, lr_c3))
              
    is_terminal = progress >= p_c3
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # --- 2. Cyclic Alpha (Mid-Run Feasibility Bursts) ---
    # To take full advantage of the LR warm restarts, we drop the penalty 
    # to a fraction at the start of each cycle (allowing free geometric movement) 
    # and ramp it up sharply at the end of each cycle (restoring feasibility).
    # This prevents the solver from getting wedged in invalid configurations.
    
    def cyclic_alpha(p, p_start, p_end, alpha_base, alpha_peak):
        p_norm = jnp.clip((p - p_start) / (p_end - p_start), 0.0, 1.0)
        # Sharp exponential-like ramp at the very end of the cycle
        return alpha_base + (alpha_peak - alpha_base) * (p_norm ** 4)
        
    a_c1 = cyclic_alpha(progress, 0.00, p_c1, alpha0 * 0.05, alpha0 * 5.0)
    a_c2 = cyclic_alpha(progress, p_c1, p_c2, alpha0 * 0.20, alpha0 * 15.0)
    a_c3 = cyclic_alpha(progress, p_c2, p_c3, alpha0 * 1.00, alpha0 * 30.0)
    
    alpha_main = jnp.where(progress < p_c1, a_c1,
                 jnp.where(progress < p_c2, a_c2, a_c3))
                 
    # Terminal absolute compliance
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # --- 3. Synchronized Moments ---
    # As alpha bursts upward, beta2 must rise simultaneously to absorb the stiff 
    # penalty gradients, and momentum drops to prevent overshoot.
    
    def cyclic_beta(p, p_start, p_end, b_start, b_end):
        p_norm = jnp.clip((p - p_start) / (p_end - p_start), 0.0, 1.0)
        return b_start + (b_end - b_start) * (p_norm ** 2)
        
    b1_c1 = cyclic_beta(progress, 0.00, p_c1, 0.20, 0.05)
    b1_c2 = cyclic_beta(progress, p_c1, p_c2, 0.15, 0.03)
    b1_c3 = cyclic_beta(progress, p_c2, p_c3, 0.10, 0.01)
    
    beta1_main = jnp.where(progress < p_c1, b1_c1,
                 jnp.where(progress < p_c2, b1_c2, b1_c3))
                 
    b2_c1 = cyclic_beta(progress, 0.00, p_c1, 0.10, 0.60)
    b2_c2 = cyclic_beta(progress, p_c1, p_c2, 0.20, 0.85)
    b2_c3 = cyclic_beta(progress, p_c2, p_c3, 0.40, 0.95)
    
    beta2_main = jnp.where(progress < p_c1, b2_c1,
                 jnp.where(progress < p_c2, b2_c2, b2_c3))
                 
    # Zero momentum in the terminal phase to kill any lingering oscillations
    beta1 = jnp.where(is_terminal, 0.00, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2