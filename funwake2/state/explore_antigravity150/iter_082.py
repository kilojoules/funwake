import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- Cyclic SGDR with Warm Restarts and Anti-Cyclic Penalty ---
    # Structurally different from WSD/Cosine: We use multi-cycle cosine annealing.
    # Each cycle explores at a high LR (and low penalty), then refines at a low LR (high penalty).
    # This prevents the optimizer from getting trapped early on, allowing it to hop 
    # out of local minima during the warm restarts.
    
    p1, p2, p3 = 0.20, 0.55, 0.92
    
    # Progress within the current cycle (0.0 to 1.0)
    c_prog = jnp.where(progress < p1, progress / p1,
               jnp.where(progress < p2, (progress - p1) / (p2 - p1),
                 jnp.where(progress < p3, (progress - p2) / (p3 - p2), 1.0)))
                 
    # Envelope maximums for Learning Rate (decaying over cycles)
    lr_max_c = jnp.where(progress < p1, 1.4 * D_f,
                 jnp.where(progress < p2, 0.7 * D_f,
                   jnp.where(progress < p3, 0.3 * D_f, gamma_min_f)))
                   
    # Envelope for Alpha (Penalty grows progressively stricter across cycles)
    alpha_max_c = jnp.where(progress < p1, alpha0 * 2.0,
                    jnp.where(progress < p2, alpha0 * 8.0,
                      jnp.where(progress < p3, alpha0 * 20.0, alpha0 * 20.0)))
                      
    alpha_min_c = jnp.where(progress < p1, alpha0 * 0.1,
                    jnp.where(progress < p2, alpha0 * 0.5,
                      jnp.where(progress < p3, alpha0 * 1.0, alpha0 * 1.0)))
                      
    # Cosine multiplier: 1.0 at start of cycle, 0.0 at end of cycle
    cos_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * c_prog))
    
    # Cyclic LR: high at start of cycle, decays to gamma_min
    lr_main = gamma_min_f + (lr_max_c - gamma_min_f) * cos_mult
    
    # Anti-cyclic Alpha: low when LR is high (explore), high when LR is low (refine)
    alpha_main = alpha_min_c + (alpha_max_c - alpha_min_c) * (1.0 - cos_mult)
    
    # Phase-Transition Adam Moments
    # High momentum (low beta2) when exploring; damped momentum (high beta2) when refining
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.15, 0.90
    
    beta1_main = b1_end + (b1_start - b1_end) * cos_mult
    beta2_main = b2_end + (b2_start - b2_end) * cos_mult
    
    # --- Terminal Feasibility Restoration Spike ---
    # Strict compliance guaranteed in the final phase
    is_terminal = progress >= p3
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_term = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_term, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2